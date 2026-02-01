import Logging

public extension ManasConformanceRunner {
    mutating func runWithLogging(
        implementationId: String,
        version: String,
        badges: [Badge],
        logger: Logger? = nil
    ) throws -> (ConformanceSuiteReport, ConformanceReport) {
        let activeLogger = logger ?? ManasRuntime().logger
        activeLogger.info("Conformance run started", metadata: [
            "implementationId": "\(implementationId)",
            "version": "\(version)",
            "badges": "\(badges.map { $0.rawValue }.joined(separator: ","))"
        ])
        let result = try run(implementationId: implementationId, version: version, badges: badges)
        activeLogger.info("Conformance run completed", metadata: [
            "suiteVersion": "\(suiteVersion)",
            "passed": "\(result.0.summary.passes)"
        ])
        return result
    }
}
