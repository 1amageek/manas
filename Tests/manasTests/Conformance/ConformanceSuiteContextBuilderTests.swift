import Foundation
import Testing
@testable import manas

@Test func conformanceSuiteContextBuilderHashesConfig() async throws {
    let physical = try PhysicalBounds(
        maxAngularRate: 1.0,
        maxTiltRadians: 1.0,
        maxLinearAcceleration: 1.0
    )
    let sensor = try SensorBounds(
        noiseAmplitude: 0.1,
        biasDrift: 0.1,
        delay: 0.0,
        bandwidthHz: 10.0
    )
    let disturbance = try DisturbanceBounds(
        torqueAmplitude: 0.1,
        torqueBandwidthHz: 1.0,
        forceAmplitude: 0.0
    )
    let actuator = try ActuatorBounds(
        saturationLimit: 1.0,
        rateLimit: 1.0,
        delay: 0.0
    )
    let rates = try UpdateRates(controllerUpdate: 0.01, sensorSample: 0.01, actuatorUpdate: 0.01)
    let oed = try OperatingEnvelope(
        id: "oed",
        version: "1.0.0",
        physical: physical,
        sensor: sensor,
        disturbance: disturbance,
        actuator: actuator,
        updateRates: rates
    )

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
        bands: [try Band(minimum: 0.1, maximum: 0.2)],
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
        duration: 1.0,
        deltaTime: 0.01,
        stepTime: 0.1,
        amplitude: amplitude,
        slope: slope,
        frequency: frequency,
        seedBase: 1,
        perturbationDelta: 0.01,
        modeInductionOffsets: [0.0],
        minimumCutoffHz: 0.5
    )

    let context = try ConformanceSuiteContextBuilder.build(
        oed: oed,
        suiteVersion: "0.1.0",
        config: config,
        coverage: coverage
    )
    #expect(context.configHash.isEmpty == false)
}
