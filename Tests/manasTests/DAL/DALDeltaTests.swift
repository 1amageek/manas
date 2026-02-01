import Foundation
import Testing
@testable import manas

private struct DeltaLearner: ActuatorLearner {
    let delta: Double

    mutating func infer(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        baseCommands: [ActuatorCommand]
    ) throws -> [ActuatorCommandDelta] {
        _ = drives
        _ = telemetry
        _ = baseCommands
        return [try ActuatorCommandDelta(index: ActuatorIndex(0), value: delta)]
    }

    mutating func update(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        deltaTime: TimeInterval
    ) throws -> LearningReport? {
        _ = drives
        _ = telemetry
        _ = deltaTime
        return try LearningReport(parameterDeltaNorm: 0, parameterDerivativeNorm: 0)
    }
}

@Test func dalAppliesLearnerDeltasBeforeSafetyFilter() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: DeltaLearner(delta: 0.4),
        safetyFilter: safety
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 0.2)
    let telemetry = DALTelemetry(motors: [])
    let commands = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)

    #expect(commands.count == 1)
    #expect(abs(commands[0].value - 0.6) < 1e-9)
}

@Test func dalClampsLearnerDeltasThroughSafetyFilter() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: DeltaLearner(delta: 2.0),
        safetyFilter: safety
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 0.2)
    let telemetry = DALTelemetry(motors: [])
    let commands = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)

    #expect(commands.count == 1)
    #expect(commands[0].value == 1.0)
}
