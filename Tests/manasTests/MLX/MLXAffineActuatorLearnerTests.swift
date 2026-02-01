import Testing
@testable import manas
import ManasMLX

@Test func affineLearnerInferClampsToDeltaMax() async throws {
    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0), ActuatorIndex(1)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.15,
        learningRate: 0.0,
        targetTelemetry: .rpm,
        enabled: false
    )
    var learner = MLXAffineActuatorLearner(configuration: config)
    let drives = [
        try DriveIntent(index: DriveIndex(0), activation: 1.0),
        try DriveIntent(index: DriveIndex(1), activation: 0.5),
    ]
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 800.0, current: nil, voltage: nil, temperature: nil, escState: nil),
        try MotorTelemetry(index: ActuatorIndex(1), rpm: 200.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])
    let baseCommands = [
        try ActuatorCommand(index: ActuatorIndex(0), value: 0.0),
        try ActuatorCommand(index: ActuatorIndex(1), value: 0.0),
    ]

    let deltas = try learner.infer(drives: drives, telemetry: telemetry, baseCommands: baseCommands)
    #expect(deltas.count == 2)
    for delta in deltas {
        #expect(abs(delta.value) <= 0.150001)
    }
}

@Test func affineLearnerMissingBaseCommandThrows() async throws {
    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0), ActuatorIndex(1)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.0,
        targetTelemetry: .rpm,
        enabled: false
    )
    var learner = MLXAffineActuatorLearner(configuration: config)
    let drives = [
        try DriveIntent(index: DriveIndex(0), activation: 1.0),
        try DriveIntent(index: DriveIndex(1), activation: 0.5),
    ]
    let telemetry = DALTelemetry(motors: [])
    let baseCommands = [
        try ActuatorCommand(index: ActuatorIndex(0), value: 0.0),
    ]

    do {
        _ = try learner.infer(drives: drives, telemetry: telemetry, baseCommands: baseCommands)
        #expect(Bool(false))
    } catch let error as MLXAffineActuatorLearner.ValidationError {
        #expect(error == .missingBaseCommand(ActuatorIndex(1)))
    }
}

@Test func affineLearnerUpdateProducesReportWhenEnabled() async throws {
    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.01,
        targetTelemetry: .rpm,
        enabled: true
    )
    var learner = MLXAffineActuatorLearner(configuration: config)
    let drives = [try DriveIntent(index: DriveIndex(0), activation: 1.0)]
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])

    let report = try learner.update(drives: drives, telemetry: telemetry, deltaTime: 0.01)
    #expect(report != nil)
    #expect(report?.parameterDeltaNorm ?? -1.0 >= 0.0)
    #expect(report?.parameterDerivativeNorm ?? -1.0 >= 0.0)
}

@Test func affineLearnerUpdateReturnsZeroWhenDisabled() async throws {
    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.01,
        targetTelemetry: .rpm,
        enabled: false
    )
    var learner = MLXAffineActuatorLearner(configuration: config)
    let drives = [try DriveIntent(index: DriveIndex(0), activation: 1.0)]
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])

    let report = try learner.update(drives: drives, telemetry: telemetry, deltaTime: 0.01)
    #expect(report?.parameterDeltaNorm == 0.0)
    #expect(report?.parameterDerivativeNorm == 0.0)
}
