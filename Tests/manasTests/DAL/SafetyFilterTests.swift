import Testing
@testable import manas

@Test func safetyFilterClampsAndRateLimits() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: 1.0)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    var filter = SafetyFilter(limits: limits)

    let first = try ActuatorCommand(index: ActuatorIndex(0), value: 2.0)
    let output1 = try filter.apply(commands: [first], deltaTime: 0.5)
    #expect(output1[0].value == 1.0)

    let second = try ActuatorCommand(index: ActuatorIndex(0), value: -1.0)
    let output2 = try filter.apply(commands: [second], deltaTime: 0.5)
    #expect(output2[0].value == 0.5)
}

@Test func safetyFilterThrowsOnMissingLimit() async throws {
    let limits = ActuatorLimits(limits: [:])
    var filter = SafetyFilter(limits: limits)
    let command = try ActuatorCommand(index: ActuatorIndex(1), value: 0.0)

    do {
        _ = try filter.apply(commands: [command], deltaTime: 0.1)
        #expect(Bool(false))
    } catch let error as SafetyFilter.ValidationError {
        #expect(error == .missingLimit(ActuatorIndex(1)))
    }
}
