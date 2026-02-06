import Foundation

public struct QualityGating: Gating {
    public struct Configuration: Sendable, Equatable {
        public let minGate: Double
        public let maxGate: Double

        public init(minGate: Double, maxGate: Double) {
            let boundedMin = min(max(minGate, 0.01), 1.0)
            let boundedMax = min(max(maxGate, 0.01), 1.0)
            self.minGate = boundedMin
            self.maxGate = max(boundedMin, boundedMax)
        }
    }

    public var configuration: Configuration

    public init(configuration: Configuration) {
        self.configuration = configuration
    }

    public mutating func apply(bundle: NerveBundleOutput, time: TimeInterval) throws -> GatedBundle {
        _ = time
        let gates = bundle.quality.map { quality in
            let clamped = min(max(quality, configuration.minGate), configuration.maxGate)
            return clamped
        }
        let gatedFeatures = zip(bundle.features, gates).map { value, gate in value * gate }
        return GatedBundle(
            features: gatedFeatures,
            fastTaps: bundle.fastTaps,
            quality: bundle.quality,
            gateFactors: gates
        )
    }
}
