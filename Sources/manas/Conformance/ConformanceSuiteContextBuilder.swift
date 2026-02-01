public struct ConformanceSuiteContextBuilder {
    public static func build(
        oed: OperatingEnvelope,
        suiteVersion: String,
        config: ConformanceSuiteConfig,
        coverage: ConformanceCoverageConfig
    ) throws -> ConformanceSuiteContext {
        let bundle = ConformanceConfigBundle(suite: config, coverage: coverage)
        let configHash = try ConfigHash.hash(bundle)
        return try ConformanceSuiteContext(
            oedId: oed.id,
            oedVersion: oed.version,
            suiteVersion: suiteVersion,
            configHash: configHash
        )
    }
}
