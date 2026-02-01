import Testing
@testable import manas

@Test func energyStateRejectsNegative() async throws {
    do {
        _ = try EnergyState(index: PerceptionIndex(0), value: -0.1)
        #expect(Bool(false))
    } catch let error as EnergyState.ValidationError {
        #expect(error == .negativeValue)
    }
}

@Test func energyStateRejectsNonFinite() async throws {
    do {
        _ = try EnergyState(index: PerceptionIndex(1), value: .nan)
        #expect(Bool(false))
    } catch let error as EnergyState.ValidationError {
        #expect(error == .nonFinite)
    }
}

@Test func energyStateAcceptsValidValue() async throws {
    let state = try EnergyState(index: PerceptionIndex(2), value: 0.5)
    #expect(state.value == 0.5)
}

