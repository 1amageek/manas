import Testing
@testable import ManasCore

@Test func consciousSummaryAcceptsRequiredFiniteFields() throws {
    let summary = try ConsciousSummary(
        salience: 0.6,
        risk: 0.2,
        uncertainty: 0.1,
        constraintPressure: 0.3,
        recoveryState: 0.8,
        timestamp: 1.25
    )

    #expect(summary.salience == 0.6)
    #expect(summary.risk == 0.2)
    #expect(summary.uncertainty == 0.1)
    #expect(summary.constraintPressure == 0.3)
    #expect(summary.recoveryState == 0.8)
    #expect(summary.timestamp == 1.25)
}

@Test func consciousSummaryRejectsNonFiniteField() throws {
    do {
        _ = try ConsciousSummary(
            salience: .nan,
            risk: 0.2,
            uncertainty: 0.1,
            constraintPressure: 0.3,
            recoveryState: 0.8,
            timestamp: 1.25
        )
        #expect(Bool(false))
    } catch let error as ConsciousSummary.ValidationError {
        #expect(error == .nonFinite("salience"))
    }
}

@Test func consciousSummaryRejectsNegativeTimestamp() throws {
    do {
        _ = try ConsciousSummary(
            salience: 0.6,
            risk: 0.2,
            uncertainty: 0.1,
            constraintPressure: 0.3,
            recoveryState: 0.8,
            timestamp: -0.01
        )
        #expect(Bool(false))
    } catch let error as ConsciousSummary.ValidationError {
        #expect(error == .invalidRange("timestamp"))
    }
}
