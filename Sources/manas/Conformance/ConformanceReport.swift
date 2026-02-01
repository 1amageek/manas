public struct ConformanceReport: Sendable, Codable, Equatable {
    public enum Status: String, Sendable, Codable {
        case pass
        case fail
    }

    public enum ValidationError: Error, Equatable {
        case empty(String)
    }

    public let implementationId: String
    public let version: String
    public let badges: [Badge]
    public let context: ConformanceSuiteContext
    public let summary: ConformanceSuiteReport.Summary
    public let status: Status

    public init(
        implementationId: String,
        version: String,
        badges: [Badge],
        context: ConformanceSuiteContext,
        summary: ConformanceSuiteReport.Summary
    ) throws {
        guard !implementationId.isEmpty else { throw ValidationError.empty("implementationId") }
        guard !version.isEmpty else { throw ValidationError.empty("version") }
        guard !badges.isEmpty else { throw ValidationError.empty("badges") }

        self.implementationId = implementationId
        self.version = version
        self.badges = badges
        self.context = context
        self.summary = summary
        self.status = summary.passes ? .pass : .fail
    }
}
