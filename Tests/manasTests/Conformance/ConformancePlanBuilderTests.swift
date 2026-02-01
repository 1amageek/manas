import Foundation
import Testing
@testable import manas

@Test func conformancePlanBuilderCreatesFamilies() async throws {
    let amplitude = try BandCoverage(
        bands: [try Band(minimum: 0.1, maximum: 0.2)],
        strategy: .midpoint
    )
    let slope = try BandCoverage(
        bands: [try Band(minimum: 0.05, maximum: 0.05)],
        strategy: .midpoint
    )
    let frequency = try BandCoverage(
        bands: [try Band(minimum: 1.0, maximum: 1.0)],
        strategy: .midpoint
    )
    let coverage = try ConformanceCoverageConfig(
        duration: 0.2,
        deltaTime: 0.05,
        stepTime: 0.1,
        amplitude: amplitude,
        slope: slope,
        frequency: frequency,
        seedBase: 42,
        perturbationDelta: 0.01,
        modeInductionOffsets: [-0.01, 0.0, 0.01],
        minimumCutoffHz: 0.5
    )

    let plan = try ConformancePlanBuilder.build(
        coverage: coverage,
        energyIndices: [PerceptionIndex(0)],
        phaseIndices: [PhaseIndex(0)]
    )

    #expect(plan.continuityPairs.count == 1)
    #expect(plan.totalVariationFamilies.count == 1)
    #expect(plan.snappingFamilies.count > 0)
    #expect(plan.phaseFamilies.count > 0)
    #expect(plan.modeInductionFamilies.count == 1)
}

