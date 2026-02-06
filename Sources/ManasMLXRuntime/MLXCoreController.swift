import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct ManasMLXCoreController: CoreController {
    public var model: ManasMLXCore
    public var state: ManasMLXCoreState?
    public var activationRange: ClosedRange<Double>

    public init(
        model: ManasMLXCore,
        state: ManasMLXCoreState? = nil,
        activationRange: ClosedRange<Double> = -1.0...1.0
    ) {
        self.model = model
        self.state = state
        self.activationRange = activationRange
    }

    public mutating func update(trunks: TrunkBundle, time: TimeInterval) throws -> [DriveIntent] {
        let vector = concatTrunks(trunks)
        let input = MLXArray(converting: vector.map(Double.init), [1, 1, vector.count])
        let output = model.forward(trunks: input, state: state)
        state = output.nextState
        let lastIndex = output.drives.dim(-2) - 1
        let last = output.drives[MLXEllipsisIndex.ellipsis, lastIndex, 0...]
        eval(last)
        let values = last.asArray(Float.self)
        return values.enumerated().compactMap { index, value in
            let clamped = min(max(Double(value), activationRange.lowerBound), activationRange.upperBound)
            return try? DriveIntent(index: DriveIndex(UInt32(index)), activation: clamped)
        }
    }

    private func concatTrunks(_ bundle: TrunkBundle) -> [Float] {
        bundle.energy.map(Float.init)
        + bundle.phase.map(Float.init)
        + bundle.quality.map(Float.init)
        + bundle.spike.map(Float.init)
    }
}
