import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct ManasMLXReflexController: ReflexController {
    public var model: ManasMLXReflex
    public var deltaRange: ClosedRange<Double>

    public init(model: ManasMLXReflex, deltaRange: ClosedRange<Double> = -1.0...1.0) {
        self.model = model
        self.deltaRange = deltaRange
    }

    public mutating func update(
        bundle: NerveBundleOutput,
        trunks: TrunkBundle,
        time: TimeInterval
    ) throws -> [ReflexCorrection] {
        let input = MLXArray(
            converting: bundle.fastTaps,
            [1, bundle.fastTaps.count]
        )
        let output = model.forward(input)
        eval(output.clamp)
        eval(output.damping)
        eval(output.delta)

        let clamp = output.clamp.asArray(Float.self)
        let damping = output.damping.asArray(Float.self)
        let delta = output.delta.asArray(Float.self)
        let driveCount = model.config.driveCount

        return (0..<driveCount).map { index in
            let c = clampValue(Double(clamp[safe: index] ?? 1.0), min: 0.0, max: 1.0)
            let d = clampValue(Double(damping[safe: index] ?? 0.0), min: 0.0, max: 1.0)
            let raw = Double(delta[safe: index] ?? 0.0)
            let t = clampValue(raw, min: deltaRange.lowerBound, max: deltaRange.upperBound)
            return try? ReflexCorrection(
                driveIndex: DriveIndex(UInt32(index)),
                clampMultiplier: c,
                damping: d,
                delta: t
            )
        }.compactMap { $0 }
    }

    private func clampValue(_ value: Double, min: Double, max: Double) -> Double {
        Swift.max(min, Swift.min(max, value))
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard index >= 0 && index < count else { return nil }
        return self[index]
    }
}
