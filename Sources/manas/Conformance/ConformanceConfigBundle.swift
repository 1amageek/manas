public struct ConformanceConfigBundle: Sendable, Codable, Equatable {
    public let suite: ConformanceSuiteConfig
    public let coverage: ConformanceCoverageConfig

    public init(suite: ConformanceSuiteConfig, coverage: ConformanceCoverageConfig) {
        self.suite = suite
        self.coverage = coverage
    }
}

