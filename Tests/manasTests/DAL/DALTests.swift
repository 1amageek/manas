import Testing
@testable import manas

@Test func dalAppliesMapperLearningAndSafety() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: NoActuatorLearning(),
        safetyFilter: safety
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 2.0)
    let telemetry = DALTelemetry(motors: [])
    let commands = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)
    #expect(commands.count == 1)
    #expect(commands[0].value == 1.0)
}
