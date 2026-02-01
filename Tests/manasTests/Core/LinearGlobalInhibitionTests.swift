import Testing
@testable import manas

@Test func linearGlobalInhibitionScales() async throws {
    let model = LinearGlobalInhibition()
    let factorAtZero = try model.factor(totalEnergy: 0.0, existThreshold: 10.0)
    #expect(factorAtZero == 1.0)

    let factorMid = try model.factor(totalEnergy: 5.0, existThreshold: 10.0)
    #expect(factorMid == 0.5)

    let factorOver = try model.factor(totalEnergy: 15.0, existThreshold: 10.0)
    #expect(factorOver == 0.0)
}

