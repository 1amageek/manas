import Foundation
import Testing
@testable import manas

@Test func inputFamilyValidatorRejectsOutOfRangeEnergy() async throws {
    let energyMap = try NormalizationMap<PerceptionIndex>(ranges: [PerceptionIndex(0): 1.0])
    let phaseMap = try NormalizationMap<PhaseIndex>(ranges: [PhaseIndex(0): 1.0])

    let energyChannel = SignalChannel(
        index: PerceptionIndex(0),
        generator: AnySignalGenerator(try StepSignal(initialValue: 2.0, stepValue: 2.0, stepTime: 0.0))
    )
    let phaseChannel = SignalChannel(
        index: PhaseIndex(0),
        generator: AnySignalGenerator(try StepSignal(initialValue: 0.0, stepValue: 0.0, stepTime: 0.0))
    )
    let family = try InputFamily(
        duration: 0.1,
        deltaTime: 0.05,
        energyChannels: [energyChannel],
        phaseChannels: [phaseChannel]
    )

    do {
        try InputFamilyValidator.validate(family: family, energyRanges: energyMap, phaseRanges: phaseMap)
        #expect(Bool(false))
    } catch let error as InputFamilyValidator.ValidationError {
        #expect(error == .energyOutOfRange(PerceptionIndex(0), 2.0, 1.0))
    }
}
