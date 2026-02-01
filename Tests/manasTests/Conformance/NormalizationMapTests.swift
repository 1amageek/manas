import Testing
@testable import manas

@Test func normalizationMapRejectsNonPositiveRange() async throws {
    do {
        _ = try NormalizationMap<PerceptionIndex>(ranges: [PerceptionIndex(0): 0.0])
        #expect(Bool(false))
    } catch let error as NormalizationMap<PerceptionIndex>.ValidationError {
        #expect(error == .nonPositive(PerceptionIndex(0)))
    }
}

@Test func normalizationMapBuildsVector() async throws {
    let map = try NormalizationMap<PerceptionIndex>(ranges: [
        PerceptionIndex(1): 2.0,
        PerceptionIndex(0): 1.0,
    ])
    let vector = try map.normalizedVector(values: [
        PerceptionIndex(0): 1.0,
        PerceptionIndex(1): 2.0,
    ])
    #expect(vector == [1.0, 1.0])
}

