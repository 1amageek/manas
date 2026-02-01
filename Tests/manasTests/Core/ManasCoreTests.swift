import Testing
@testable import manas

@Test func manasCoreComputesReflexesAndRegime() async throws {
    let weights = try EnergyWeights(weights: [PerceptionIndex(0): 1.0])
    let thresholds = [
        PerceptionIndex(0): try PerceptionThresholds(
            localEnergy: 1.0,
            criticalGradient: 1.0,
            reflexEnergy: 2.0,
            reflexGradient: 2.0
        ),
    ]
    let globals = try GlobalThresholds(existEnergy: 5.0)
    let limits = try DriveLimits(limits: [DriveIndex(0): -1.0...1.0])
    let rates = try UpdateRates(controllerUpdate: 0.01, sensorSample: 0.01, actuatorUpdate: 0.01)
    let params = try ManasParameters(
        weights: weights,
        thresholds: thresholds,
        globalThresholds: globals,
        driveLimits: limits,
        updatePeriod: rates
    )

    var core = ManasCore(
        parameters: params,
        driveSynthesizer: ZeroDriveSynthesizer(driveLimits: limits),
        globalInhibition: LinearGlobalInhibition(),
        localInhibition: LinearLocalInhibitionModel(),
        coupling: try InhibitionCoupling(matrix: [:]),
        reflexPolicy: ZeroReflexOverride(driveLimits: limits),
        gradientEstimator: FiniteDifferenceEnergyGradientEstimator()
    )

    let energy = try EnergyState(index: PerceptionIndex(0), value: 3.0)
    let output = try core.update(energies: [energy], phases: [], deltaTime: 0.01)
    #expect(output.regime == .normal)
    #expect(output.reflexes == [PerceptionIndex(0)])
}
