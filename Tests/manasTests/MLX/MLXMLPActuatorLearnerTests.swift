import Testing
@testable import manas
import ManasMLX

@Test func mlpLearnerInferClampsToDeltaMax() async throws {
    let config = try MLXMLPActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0), ActuatorIndex(1)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.0,
        hiddenSize: 8,
        targetTelemetry: .rpm,
        enabled: false
    )
    var learner = MLXMLPActuatorLearner(configuration: config)
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
        #expect(abs(delta.value) <= 0.200001)
    }
}

@Test func mlpLearnerMissingBaseCommandThrows() async throws {
    let config = try MLXMLPActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0), ActuatorIndex(1)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.0,
        hiddenSize: 8,
        targetTelemetry: .rpm,
        enabled: false
    )
    var learner = MLXMLPActuatorLearner(configuration: config)
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
    } catch let error as MLXMLPActuatorLearner.ValidationError {
        #expect(error == .missingBaseCommand(ActuatorIndex(1)))
    }
}

@Test func mlpLearnerUpdateProducesReportWhenEnabled() async throws {
    let config = try MLXMLPActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.01,
        hiddenSize: 8,
        targetTelemetry: .rpm,
        enabled: true
    )
    var learner = MLXMLPActuatorLearner(configuration: config)
    let drives = [try DriveIntent(index: DriveIndex(0), activation: 1.0)]
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])

    let report = try learner.update(drives: drives, telemetry: telemetry, deltaTime: 0.01)
    #expect(report != nil)
    #expect(report?.parameterDeltaNorm ?? -1.0 >= 0.0)
    #expect(report?.parameterDerivativeNorm ?? -1.0 >= 0.0)
}

@Test func mlpLearnerUpdateReturnsZeroWhenDisabled() async throws {
    let config = try MLXMLPActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.01,
        hiddenSize: 8,
        targetTelemetry: .rpm,
        enabled: false
    )
    var learner = MLXMLPActuatorLearner(configuration: config)
    let drives = [try DriveIntent(index: DriveIndex(0), activation: 1.0)]
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 0.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])

    let report = try learner.update(drives: drives, telemetry: telemetry, deltaTime: 0.01)
    #expect(report?.parameterDeltaNorm == 0.0)
    #expect(report?.parameterDerivativeNorm == 0.0)
}
