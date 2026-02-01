import Testing
@testable import manas

@Test func operatingEnvelopeRejectsEmptyId() async throws {
    let physical = try PhysicalBounds(
        maxAngularRate: 1.0,
        maxTiltRadians: 1.0,
        maxLinearAcceleration: 1.0
    )
    let sensor = try SensorBounds(
        noiseAmplitude: 0.1,
        biasDrift: 0.1,
        delay: 0.0,
        bandwidthHz: 100.0
    )
    let disturbance = try DisturbanceBounds(
        torqueAmplitude: 0.1,
        torqueBandwidthHz: 10.0,
        forceAmplitude: 0.0
    )
    let actuator = try ActuatorBounds(
        saturationLimit: 1.0,
        rateLimit: 1.0,
        delay: 0.0
    )
    let rates = try UpdateRates(
        controllerUpdate: 0.01,
        sensorSample: 0.01,
        actuatorUpdate: 0.01
    )

    do {
        _ = try OperatingEnvelope(
            id: "",
            version: "1.0.0",
            physical: physical,
            sensor: sensor,
            disturbance: disturbance,
            actuator: actuator,
            updateRates: rates
        )
        #expect(Bool(false))
    } catch let error as OperatingEnvelope.ValidationError {
        #expect(error == .emptyIdentifier)
    }
}

@Test func updateRatesRejectsNonPositive() async throws {
    do {
        _ = try UpdateRates(
            controllerUpdate: 0.0,
            sensorSample: 0.01,
            actuatorUpdate: 0.01
        )
        #expect(Bool(false))
    } catch let error as UpdateRates.ValidationError {
        #expect(error == .nonPositive("controllerUpdate"))
    }
}
