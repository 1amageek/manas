import Foundation
import Testing
@testable import manas

private struct ZeroTarget: ManasConformanceTarget {
    mutating func reset() {}

    mutating func step(
        energies: [EnergyState],
        phases: [PhaseState],
        deltaTime: TimeInterval
    ) throws -> [DriveIntent] {
        _ = energies
        _ = phases
        _ = deltaTime
        return [try DriveIntent(index: DriveIndex(0), activation: 0.0)]
    }
}

@Test func conformanceSuiteRunsAllSections() async throws {
    let energyMap = try NormalizationMap<PerceptionIndex>(ranges: [PerceptionIndex(0): 1.0])
    let phaseMap = try NormalizationMap<PhaseIndex>(ranges: [PhaseIndex(0): 1.0])
    let driveMap = try NormalizationMap<DriveIndex>(ranges: [DriveIndex(0): 1.0])
    let bundle = NormalizationBundle(energy: energyMap, phase: phaseMap, drive: driveMap)

    let config = try ConformanceSuiteConfig(
        l2: 1.0,
        lInf: 1.0,
        totalVariationLimit: 1.0,
        snappingEpsilon: 0.001,
        snappingMaxClusters: 2,
        minimumPhaseVariance: 0.0,
        phaseBandwidthHz: 10.0,
        phaseSnappingEpsilon: 0.001,
        phaseSnappingMaxClusters: 2,
        modeInductionEpsilon: 0.01,
        modeInductionMaxModes: 1,
        steadyWindowSize: 2
    )

    var runner = ConformanceRunner(target: ZeroTarget(), normalization: bundle)
    var suite = ConformanceSuite(runner: runner, config: config)

    let stepConfig = try StepFamilyConfig(
        initialValue: 0.0,
        stepValue: 0.0,
        stepTime: 0.0,
        duration: 0.1,
        deltaTime: 0.05
    )
    let base = try InputFamilyFactory.stepFamily(
        config: stepConfig,
        energyIndices: [PerceptionIndex(0)],
        phaseIndices: [PhaseIndex(0)]
    )
    let perturbed = try InputFamilyFactory.stepFamily(
        config: stepConfig,
        energyIndices: [PerceptionIndex(0)],
        phaseIndices: [PhaseIndex(0)]
    )

    let report = try suite.run(
        continuityPairs: [ConformanceInputPair(base: base, perturbed: perturbed)],
        totalVariationFamilies: [base],
        snappingFamilies: [base],
        phaseVarianceFamilies: [base],
        modeInductionFamilies: [[base]]
    )

    #expect(report.continuity.count == 1)
    #expect(report.totalVariation.count == 1)
    #expect(report.snapping.count == 1)
    #expect(report.phaseVariance.count == 1)
    #expect(report.phaseBandwidth.count == 1)
    #expect(report.phaseSnapping.count == 1)
    #expect(report.modeInduction.count == 1)
    #expect(report.summary.continuityPasses == true)
}
