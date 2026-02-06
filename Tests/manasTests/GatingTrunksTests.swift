import Testing
@testable import ManasCore

@Test func qualityGatingAppliesToFeaturesOnly() async throws {
    let bundle = NerveBundleOutput(
        features: [1.0, -1.0],
        fastTaps: [0.5, -0.5],
        quality: [0.2, 0.8]
    )
    var gating = QualityGating(configuration: .init(minGate: 0.2, maxGate: 1.0))
    let gated = try gating.apply(bundle: bundle, time: 0.0)

    #expect(gated.features[0] == 0.2)
    #expect(gated.features[1] == -0.8)
    #expect(gated.fastTaps == bundle.fastTaps)
}

@Test func qualityGatingRespectsMinGateFloor() async throws {
    let bundle = NerveBundleOutput(
        features: [1.0],
        fastTaps: [0.0],
        quality: [0.0]
    )
    var gating = QualityGating(configuration: .init(minGate: 0.0, maxGate: 1.0))
    let gated = try gating.apply(bundle: bundle, time: 0.0)
    #expect(abs(gated.gateFactors[0] - 0.01) < 1.0e-9)
    #expect(abs(gated.features[0] - 0.01) < 1.0e-9)
}

@Test func qualityGatingClampsWithinConfiguredRange() async throws {
    let bundle = NerveBundleOutput(
        features: [1.0, 1.0],
        fastTaps: [0.2, -0.2],
        quality: [2.0, 0.0]
    )
    var gating = QualityGating(configuration: .init(minGate: 0.2, maxGate: 0.8))
    let gated = try gating.apply(bundle: bundle, time: 0.0)
    #expect(gated.gateFactors == [0.8, 0.2])
    #expect(gated.fastTaps == bundle.fastTaps)
    #expect(gated.quality == bundle.quality)
}

@Test func identityGatingLeavesGateFactorsAtOne() async throws {
    let bundle = NerveBundleOutput(
        features: [0.5, -0.5],
        fastTaps: [0.25, -0.25],
        quality: [0.4, 0.6]
    )
    var gating = IdentityGating()
    let gated = try gating.apply(bundle: bundle, time: 0.0)
    #expect(gated.features == bundle.features)
    #expect(gated.fastTaps == bundle.fastTaps)
    #expect(gated.quality == bundle.quality)
    #expect(gated.gateFactors == [1.0, 1.0])
}

@Test func basicTrunksBuilderMapsEnergyPhaseQualityAndSpike() async throws {
    var builder = BasicTrunksBuilder()
    let gated = GatedBundle(features: [-0.5, 0.25], fastTaps: [0.2, -0.1], quality: [0.4, 0.9], gateFactors: [1.0, 1.0])
    let trunks = try builder.build(from: gated, time: 0.0)
    #expect(trunks.energy == [0.5, 0.25])
    #expect(trunks.phase == gated.features)
    #expect(trunks.quality == gated.quality)
    #expect(trunks.spike == [0.2, 0.1])
}

@Test func spikeTrunksDetectsChanges() async throws {
    var builder = SpikeTrunksBuilder(configuration: .init(spikeGain: 1.0))
    let gated = GatedBundle(features: [0.2], fastTaps: [0.1], quality: [1.0], gateFactors: [1.0])
    _ = try builder.build(from: gated, time: 0.0)
    let next = GatedBundle(features: [0.2], fastTaps: [0.4], quality: [1.0], gateFactors: [1.0])
    let trunks = try builder.build(from: next, time: 0.01)
    #expect(abs(trunks.spike[0] - 0.3) < 1e-9)
}
