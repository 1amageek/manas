import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct ManasMLXReflexController: ReflexController {
    public enum ControllerError: Error, Equatable {
        case outputSizeMismatch(head: String, expected: Int, actual: Int)
        case nonFiniteOutput(head: String, index: Int)
    }

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

        guard clamp.count >= driveCount else {
            throw ControllerError.outputSizeMismatch(head: "clamp", expected: driveCount, actual: clamp.count)
        }
        guard damping.count >= driveCount else {
            throw ControllerError.outputSizeMismatch(head: "damping", expected: driveCount, actual: damping.count)
        }
        guard delta.count >= driveCount else {
            throw ControllerError.outputSizeMismatch(head: "delta", expected: driveCount, actual: delta.count)
        }

        return try (0..<driveCount).map { index in
            let cRaw = Double(clamp[index])
            let dRaw = Double(damping[index])
            let raw = Double(delta[index])
            guard cRaw.isFinite else {
                throw ControllerError.nonFiniteOutput(head: "clamp", index: index)
            }
            guard dRaw.isFinite else {
                throw ControllerError.nonFiniteOutput(head: "damping", index: index)
            }
            guard raw.isFinite else {
                throw ControllerError.nonFiniteOutput(head: "delta", index: index)
            }
            let c = clampValue(cRaw, min: 0.0, max: 1.0)
            let d = clampValue(dRaw, min: 0.0, max: 1.0)
            let t = clampValue(raw, min: deltaRange.lowerBound, max: deltaRange.upperBound)
            return try ReflexCorrection(
                driveIndex: DriveIndex(UInt32(index)),
                clampMultiplier: c,
                damping: d,
                delta: t
            )
        }
    }

    private func clampValue(_ value: Double, min: Double, max: Double) -> Double {
        Swift.max(min, Swift.min(max, value))
    }
}
