import Testing
@testable import manas

@Test func linearLocalInhibitionHonorsThresholds() async throws {
    let thresholds = try PerceptionThresholds(
        localEnergy: 1.0,
        criticalGradient: 0.5,
        reflexEnergy: 2.0,
        reflexGradient: 1.0
    )
    let model = LinearLocalInhibitionModel()

    let noInhibit = try model.factor(energy: 0.5, gradient: 0.2, thresholds: thresholds)
    #expect(noInhibit == 1.0)

    let fullInhibitEnergy = try model.factor(energy: 3.0, gradient: 0.2, thresholds: thresholds)
    #expect(fullInhibitEnergy == 0.0)

    let fullInhibitGradient = try model.factor(energy: 0.5, gradient: 2.0, thresholds: thresholds)
    #expect(fullInhibitGradient == 0.0)
}

