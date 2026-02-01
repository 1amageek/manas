import Testing
@testable import manas

@Test func conformanceReportBuilderRequiresContext() async throws {
    let report = ConformanceSuiteReport(
        continuity: [],
        totalVariation: [],
        snapping: [],
        phaseVariance: [],
        phaseBandwidth: [],
        phaseSnapping: [],
        modeInduction: [],
        context: nil
    )

    do {
        _ = try ConformanceReportBuilder.build(
            implementationId: "impl",
            version: "1.0.0",
            badges: [.b0Baseline],
            report: report
        )
        #expect(Bool(false))
    } catch let error as ConformanceReportBuilder.ValidationError {
        #expect(error == .missingContext)
    }
}

