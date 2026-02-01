public struct ConformanceReportBuilder {
    public enum ValidationError: Error, Equatable {
        case missingContext
    }

    public static func build(
        implementationId: String,
        version: String,
        badges: [Badge],
        report: ConformanceSuiteReport
    ) throws -> ConformanceReport {
        guard let context = report.context else {
            throw ValidationError.missingContext
        }

        return try ConformanceReport(
            implementationId: implementationId,
            version: version,
            badges: badges,
            context: context,
            summary: report.summary
        )
    }
}

