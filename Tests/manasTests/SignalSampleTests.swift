import Testing
@testable import ManasCore

@Test func signalSampleRejectsNegativeTimestamp() async throws {
    #expect(throws: SignalSample.ValidationError.negativeTimestamp) {
        _ = try SignalSample(channelIndex: 0, value: 0.1, timestamp: -0.01)
    }
}

@Test func signalSampleRejectsNonFiniteValues() async throws {
    #expect(throws: SignalSample.ValidationError.nonFinite) {
        _ = try SignalSample(channelIndex: 0, value: Double.nan, timestamp: 0.0)
    }
    #expect(throws: SignalSample.ValidationError.nonFinite) {
        _ = try SignalSample(channelIndex: 0, value: 0.0, timestamp: Double.infinity)
    }
}
