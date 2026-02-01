import Testing
@testable import manas

@Test func conformanceSuiteValidatorDetectsDeltaMismatch() async throws {
    var energyChannel = SignalChannel(
        index: PerceptionIndex(0),
        generator: AnySignalGenerator(try StepSignal(initialValue: 0.0, stepValue: 0.0, stepTime: 0.0))
    )
    var phaseChannel = SignalChannel(
        index: PhaseIndex(0),
        generator: AnySignalGenerator(try StepSignal(initialValue: 0.0, stepValue: 0.0, stepTime: 0.0))
    )
    let family = try InputFamily(
        duration: 1.0,
        deltaTime: 0.1,
        energyChannels: [energyChannel],
        phaseChannels: [phaseChannel]
    )
    let rates = try UpdateRates(controllerUpdate: 0.05, sensorSample: 0.05, actuatorUpdate: 0.05)

    do {
        try ConformanceSuiteValidator.validateDeltaTime(
            families: [family],
            updateRates: rates,
            tolerance: 0.0
        )
        #expect(Bool(false))
    } catch let error as ConformanceSuiteValidator.ValidationError {
        #expect(error == .deltaTimeMismatch(0.1, 0.05))
    }
}

