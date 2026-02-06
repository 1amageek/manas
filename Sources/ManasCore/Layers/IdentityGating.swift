import Foundation

public struct IdentityGating: Gating {
    public init() {}

    public mutating func apply(bundle: NerveBundleOutput, time: TimeInterval) throws -> GatedBundle {
        _ = time
        let gates = Array(repeating: 1.0, count: bundle.features.count)
        return GatedBundle(
            features: bundle.features,
            fastTaps: bundle.fastTaps,
            quality: bundle.quality,
            gateFactors: gates
        )
    }
}
