import Foundation
import Testing
@testable import manas

private struct ConstantTarget: ManasConformanceTarget {
    mutating func reset() {}

    mutating func step(
        energies: [EnergyState],
        phases: [PhaseState],
        deltaTime: TimeInterval
    ) throws -> [DriveIntent] {
        _ = energies
        _ = phases
        _ = deltaTime
        return [try DriveIntent(index: DriveIndex(0), activation: 0.5)]
    }
}

@Test func conformanceRunnerComputesContinuity() async throws {
    let energyMap = try NormalizationMap<PerceptionIndex>(ranges: [PerceptionIndex(0): 1.0])
    let phaseMap = try NormalizationMap<PhaseIndex>(ranges: [PhaseIndex(0): 1.0])
    let driveMap = try NormalizationMap<DriveIndex>(ranges: [DriveIndex(0): 1.0])
    let bundle = NormalizationBundle(energy: energyMap, phase: phaseMap, drive: driveMap)

    let target = ConstantTarget()
    var runner = ConformanceRunner(target: target, normalization: bundle)

    let energyChannel = SignalChannel(
        index: PerceptionIndex(0),
        generator: AnySignalGenerator(try StepSignal(initialValue: 0.0, stepValue: 0.0, stepTime: 0.0))
    )
    let phaseChannel = SignalChannel(
        index: PhaseIndex(0),
        generator: AnySignalGenerator(try StepSignal(initialValue: 0.0, stepValue: 0.0, stepTime: 0.0))
    )
    var base = try InputFamily(
        duration: 0.1,
        deltaTime: 0.05,
        energyChannels: [energyChannel],
        phaseChannels: [phaseChannel]
    )
    var perturbed = try InputFamily(
        duration: 0.1,
        deltaTime: 0.05,
        energyChannels: [energyChannel],
        phaseChannels: [phaseChannel]
    )

    let baseRun = try runner.run(family: &base)
    let perturbedRun = try runner.run(family: &perturbed)
    let results = try runner.continuity(
        base: baseRun,
        perturbed: perturbedRun,
        l2: 1.0,
        lInf: 1.0
    )
    #expect(results.allSatisfy { $0.passes })
}
