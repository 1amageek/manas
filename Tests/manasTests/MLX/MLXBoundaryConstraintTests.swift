import Testing
@testable import manas
import ManasMLX

@Test func mlxAffineBoundaryConstraintsAllowZeroDelta() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    let constraints = try LearningConstraints(
        minUpdatePeriod: 0.001,
        maxParameterDeltaNorm: 0.0,
        maxParameterDerivativeNorm: 0.0
    )

    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.0,
        targetTelemetry: .rpm,
        enabled: true
    )
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: MLXAffineActuatorLearner(configuration: config),
        safetyFilter: safety,
        learningConstraints: constraints
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 1.0)
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])

    let commands = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)
    #expect(commands.count == 1)
}

@Test func mlxMLPBoundaryConstraintsAllowZeroDelta() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    let constraints = try LearningConstraints(
        minUpdatePeriod: 0.001,
        maxParameterDeltaNorm: 0.0,
        maxParameterDerivativeNorm: 0.0
    )

    let config = try MLXMLPActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.0,
        hiddenSize: 8,
        targetTelemetry: .rpm,
        enabled: true
    )
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: MLXMLPActuatorLearner(configuration: config),
        safetyFilter: safety,
        learningConstraints: constraints
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 1.0)
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])

    let commands = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)
    #expect(commands.count == 1)
}
