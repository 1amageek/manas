import Foundation
import Testing
@testable import manas
import ManasMLX

@Test(.timeLimit(.minutes(1))) func mlxAffineInferBenchmark() async throws {
    let config = try MLXAffineActuatorLearner.Configuration(
        actuatorIndices: [ActuatorIndex(0)],
        telemetryFeatures: [.rpm],
        driveRange: 0.0...1.0,
        telemetryRanges: [.rpm: 0.0...1000.0],
        deltaMax: 0.2,
        learningRate: 0.0,
        targetTelemetry: .rpm,
        enabled: false
    )
    var learner = MLXAffineActuatorLearner(configuration: config)
    let drives = [try DriveIntent(index: DriveIndex(0), activation: 0.7)]
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 500.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])
    let baseCommands = [try ActuatorCommand(index: ActuatorIndex(0), value: 0.0)]

    let iterations = 2000
    let start = Date()
    for _ in 0..<iterations {
        _ = try learner.infer(drives: drives, telemetry: telemetry, baseCommands: baseCommands)
    }
    let seconds = Date().timeIntervalSince(start)
    let perSecond = Double(iterations) / max(seconds, 1e-9)
    print("[Benchmark] MLXAffine infer iterations=\(iterations) elapsed=\(seconds)s iter/s=\(perSecond)")
}

@Test(.timeLimit(.minutes(1))) func mlxAffineUpdateBenchmark() async throws {
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
    let drives = [try DriveIntent(index: DriveIndex(0), activation: 0.7)]
    let telemetry = DALTelemetry(motors: [
        try MotorTelemetry(index: ActuatorIndex(0), rpm: 100.0, current: nil, voltage: nil, temperature: nil, escState: nil),
    ])

    let iterations = 500
    let start = Date()
    for _ in 0..<iterations {
        _ = try learner.update(drives: drives, telemetry: telemetry, deltaTime: 0.01)
    }
    let seconds = Date().timeIntervalSince(start)
    let perSecond = Double(iterations) / max(seconds, 1e-9)
    print("[Benchmark] MLXAffine update iterations=\(iterations) elapsed=\(seconds)s iter/s=\(perSecond)")
}
