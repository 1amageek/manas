import Testing
@testable import manas

@Test func stepSignalStepsAtTime() async throws {
    var signal = try StepSignal(initialValue: 0.0, stepValue: 1.0, stepTime: 1.0)
    let before = try signal.sample(at: 0.5)
    let after = try signal.sample(at: 1.0)
    #expect(before == 0.0)
    #expect(after == 1.0)
}

@Test func rampSignalIncreasesLinearly() async throws {
    var signal = try RampSignal(startValue: 0.0, slope: 2.0)
    let value = try signal.sample(at: 1.5)
    #expect(value == 3.0)
}

@Test func chirpSignalStartsAtZeroPhase() async throws {
    var signal = try ChirpSignal(
        amplitude: 2.0,
        initialFrequency: 1.0,
        finalFrequency: 2.0,
        duration: 1.0
    )
    let value = try signal.sample(at: 0.0)
    #expect(value == 0.0)
}

@Test func prbsSignalHoldsBetweenSwitches() async throws {
    var signal = try PRBSSignal(amplitude: 1.0, switchPeriod: 1.0, seed: 123)
    let first = try signal.sample(at: 0.0)
    let second = try signal.sample(at: 0.5)
    #expect(first == second)
}

@Test func bandLimitedNoiseIsDeterministic() async throws {
    var noiseA = try BandLimitedNoiseSignal(
        amplitude: 1.0,
        cutoffHz: 5.0,
        deltaTime: 0.01,
        seed: 42
    )
    var noiseB = try BandLimitedNoiseSignal(
        amplitude: 1.0,
        cutoffHz: 5.0,
        deltaTime: 0.01,
        seed: 42
    )

    var samplesA: [Double] = []
    var samplesB: [Double] = []
    for _ in 0..<5 {
        samplesA.append(try noiseA.sample(at: 0.0))
        samplesB.append(try noiseB.sample(at: 0.0))
    }
    #expect(samplesA == samplesB)
}

@Test func filteredPRBSIsDeterministic() async throws {
    let prbsA = try PRBSSignal(amplitude: 1.0, switchPeriod: 0.5, seed: 7)
    let prbsB = try PRBSSignal(amplitude: 1.0, switchPeriod: 0.5, seed: 7)
    var signalA = try FilteredPRBSSignal(prbs: prbsA, cutoffHz: 2.0, deltaTime: 0.01)
    var signalB = try FilteredPRBSSignal(prbs: prbsB, cutoffHz: 2.0, deltaTime: 0.01)

    var samplesA: [Double] = []
    var samplesB: [Double] = []
    for _ in 0..<5 {
        samplesA.append(try signalA.sample(at: 0.0))
        samplesB.append(try signalB.sample(at: 0.0))
    }
    #expect(samplesA == samplesB)
}

