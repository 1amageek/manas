import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct MLXTemporalCoreController: TemporalCoreController {
    public var model: ManasMLXTemporalCore
    public var state: ManasMLXCoreState?
    public var activationRange: ClosedRange<Double>
    public var intervalSeconds: Double

    public init(
        model: ManasMLXTemporalCore,
        state: ManasMLXCoreState? = nil,
        activationRange: ClosedRange<Double> = -1.0...1.0,
        intervalSeconds: Double = 0.005
    ) {
        self.model = model
        self.state = state
        self.activationRange = activationRange
        self.intervalSeconds = intervalSeconds
    }

    public mutating func updateTemporal(
        trunks: TrunkBundle,
        time: TimeInterval,
        horizonSteps: Int
    ) throws -> [TemporalDriveWindow] {
        let vector = concatTrunks(trunks)
        let input = MLXArray(converting: vector.map(Double.init), [1, 1, vector.count])

        let steps = min(horizonSteps, model.config.maxHorizonSteps)
        let output = model.forward(trunks: input, state: state, horizonSteps: steps)
        state = output.nextState

        eval(output.horizon)
        let driveCount = model.config.driveCount

        let horizonValues = output.horizon.reshaped([-1, driveCount])
        var windows: [TemporalDriveWindow] = []

        let batchSize = horizonValues.dim(0) / steps
        for b in 0..<max(batchSize, 1) {
            var stepIntents: [DriveIntent] = []
            for s in 0..<steps {
                let rowIndex = b * steps + s
                let row = horizonValues[rowIndex, 0...]
                eval(row)
                let vals = row.asArray(Float.self)
                for (d, val) in vals.enumerated() {
                    let clamped = min(max(Double(val), activationRange.lowerBound), activationRange.upperBound)
                    if let intent = try? DriveIntent(index: DriveIndex(UInt32(d)), activation: clamped) {
                        stepIntents.append(intent)
                    }
                }
            }

            let chunked = stride(from: 0, to: stepIntents.count, by: driveCount).map { start in
                Array(stepIntents[start..<min(start + driveCount, stepIntents.count)])
            }

            for chunk in chunked {
                guard !chunk.isEmpty else { continue }
            }

            if !stepIntents.isEmpty {
                let horizonIntents = stride(from: 0, to: stepIntents.count, by: driveCount).compactMap { start -> DriveIntent? in
                    guard start < stepIntents.count else { return nil }
                    return stepIntents[start]
                }
                if let window = try? TemporalDriveWindow(
                    horizon: Array(horizonIntents.prefix(steps)),
                    intervalSeconds: intervalSeconds
                ) {
                    windows.append(window)
                }
            }
        }

        return windows
    }

    private func concatTrunks(_ bundle: TrunkBundle) -> [Float] {
        bundle.energy.map(Float.init)
        + bundle.phase.map(Float.init)
        + bundle.quality.map(Float.init)
        + bundle.spike.map(Float.init)
    }
}
