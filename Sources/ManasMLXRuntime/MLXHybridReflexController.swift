import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct MLXHybridReflexController: ReflexController {
    public enum ControllerError: Error, Equatable {
        case outputSizeMismatch(expected: Int, actual: Int)
        case nonFiniteOutput(index: Int)
    }

    public var model: ManasMLXHybridReflex
    public var activationRange: ClosedRange<Double>

    public init(model: ManasMLXHybridReflex, activationRange: ClosedRange<Double> = -1.0...1.0) {
        self.model = model
        self.activationRange = activationRange
    }

    public mutating func update(
        bundle: NerveBundleOutput,
        trunks: TrunkBundle,
        time: TimeInterval
    ) throws -> [ReflexCorrection] {
        let features = MLXArray(
            converting: bundle.fastTaps,
            [1, bundle.fastTaps.count]
        )

        let omegaError = MLXArray(
            converting: trunks.spike,
            [1, trunks.spike.count]
        )

        let output = model.forward(features: features, omegaError: omegaError)
        eval(output.combined)

        let values = output.combined.asArray(Float.self)
        let actuatorCount = model.config.actuatorCount

        guard values.count >= actuatorCount else {
            throw ControllerError.outputSizeMismatch(expected: actuatorCount, actual: values.count)
        }

        return try (0..<actuatorCount).map { index in
            let raw = Double(values[index])
            guard raw.isFinite else {
                throw ControllerError.nonFiniteOutput(index: index)
            }
            let clamped = min(max(raw, activationRange.lowerBound), activationRange.upperBound)
            return try ReflexCorrection(
                driveIndex: DriveIndex(UInt32(index)),
                clampMultiplier: 1.0,
                damping: 0.0,
                delta: clamped
            )
        }
    }
}
