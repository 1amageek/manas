import Testing
@testable import manas
import ManasMLX

private typealias MLXDAL = DAL<IdentityActuatorMapper, MLXAffineActuatorLearner>

@Test func mlxLearnerExceedsParameterDeltaLimit() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    let constraints = try LearningConstraints(
        minUpdatePeriod: 0.001,
        maxParameterDeltaNorm: 0.000001,
        maxParameterDerivativeNorm: 1000.0
    )

    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 1.0,
        learningRate: 10.0,
        targetTelemetry: .rpm,
        enabled: true
    )
    var learner = MLXAffineActuatorLearner(configuration: config)
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: learner,
        safetyFilter: safety,
        learningConstraints: constraints
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 1.0)
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil)
    ])

    do {
        _ = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)
        #expect(Bool(false))
    } catch let error as MLXDAL.ValidationError {
        switch error {
        case .parameterDeltaExceeded:
            #expect(true)
        default:
            #expect(Bool(false))
        }
    }
}

@Test func mlxLearnerExceedsParameterDerivativeLimit() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    let constraints = try LearningConstraints(
        minUpdatePeriod: 0.001,
        maxParameterDeltaNorm: 1000.0,
        maxParameterDerivativeNorm: 0.000001
    )

    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 1.0,
        learningRate: 10.0,
        targetTelemetry: .rpm,
        enabled: true
    )
    var learner = MLXAffineActuatorLearner(configuration: config)
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: learner,
        safetyFilter: safety,
        learningConstraints: constraints
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 1.0)
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil)
    ])

    do {
        _ = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)
        #expect(Bool(false))
    } catch let error as MLXDAL.ValidationError {
        switch error {
        case .parameterDerivativeExceeded:
            #expect(true)
        default:
            #expect(Bool(false))
        }
    }
}
