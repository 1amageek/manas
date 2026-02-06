import Testing
@testable import ManasCore

@Test func manasStackProducesActuatorValues() async throws {
    let samples = [
        try SignalSample(channelIndex: 0, value: 0.4, timestamp: 0.0),
        try SignalSample(channelIndex: 1, value: -0.2, timestamp: 0.0),
        try SignalSample(channelIndex: 2, value: 0.1, timestamp: 0.0),
        try SignalSample(channelIndex: 3, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 4, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 5, value: 1.0, timestamp: 0.0),
    ]

    let motorNerve = ManasMotorNerve(
        mapper: IdentityMotorNerveMapper(),
        primitiveBank: try PrimitiveBankDescriptor(driveCount: 2)
    )

    var stack = ManasStack(
        bundle: Imu6NerveBundle(configuration: .init(gyroRange: -5.0...5.0, accelRange: -2.0...2.0)),
        gate: QualityGating(configuration: .init(minGate: 0.2, maxGate: 1.0)),
        trunks: SpikeTrunksBuilder(configuration: .init(spikeGain: 1.0)),
        core: SimpleCore(configuration: .init(driveCount: 2, activationRange: -1.0...1.0)),
        reflex: SimpleReflex(configuration: .init(driveCount: 2, clampSensitivity: 0.0, dampingSensitivity: 0.0, deltaGain: 0.0)),
        motorNerve: motorNerve
    )

    let telemetry = MotorNerveTelemetry(motors: [])
    let commands = try stack.update(samples: samples, time: 0.0, telemetry: telemetry, deltaTime: 0.01)

    #expect(commands.count == 2)
    for command in commands {
        #expect(command.value >= -1.0)
        #expect(command.value <= 1.0)
    }
}
