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

@Test func conformanceSuiteEngineRunsPlan() async throws {
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

    let amplitude = try BandCoverage(
        bands: [try Band(minimum: 0.1, maximum: 0.1)],
        strategy: .midpoint
    )
    let slope = try BandCoverage(
        bands: [try Band(minimum: 0.0, maximum: 0.0)],
        strategy: .midpoint
    )
    let frequency = try BandCoverage(
        bands: [try Band(minimum: 1.0, maximum: 1.0)],
        strategy: .midpoint
    )
    let coverage = try ConformanceCoverageConfig(
        duration: 0.1,
        deltaTime: 0.05,
        stepTime: 0.05,
        amplitude: amplitude,
        slope: slope,
        frequency: frequency,
        seedBase: 1,
        perturbationDelta: 0.01,
        modeInductionOffsets: [0.0],
        minimumCutoffHz: 0.5
    )

    var runner = ConformanceRunner(target: ZeroTarget(), normalization: bundle)
    var suite = ConformanceSuite(runner: runner, config: config)
    let rates = try UpdateRates(controllerUpdate: 0.05, sensorSample: 0.05, actuatorUpdate: 0.05)

    var engine = ConformanceSuiteEngine(
        suite: suite,
        coverage: coverage,
        energyIndices: bundle.energyIndices,
        phaseIndices: bundle.phaseIndices,
        updateRates: rates
    )

    let report = try engine.run()
    #expect(report.continuity.count == 1)
}
