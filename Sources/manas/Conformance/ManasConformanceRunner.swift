public struct ManasConformanceRunner<Target: ManasConformanceTarget> {
    public var engine: ConformanceSuiteEngine<Target>
    public let oed: OperatingEnvelope
    public let suiteVersion: String

    public init(
        engine: ConformanceSuiteEngine<Target>,
        oed: OperatingEnvelope,
        suiteVersion: String
    ) {
        self.engine = engine
        self.oed = oed
        self.suiteVersion = suiteVersion
    }

    public mutating func run(
        implementationId: String,
        version: String,
        badges: [Badge]
    ) throws -> (ConformanceSuiteReport, ConformanceReport) {
        let context = try ConformanceSuiteContextBuilder.build(
            oed: oed,
            suiteVersion: suiteVersion,
            config: engine.suite.config,
            coverage: engine.coverage
        )

        engine.suite = ConformanceSuite(
            runner: engine.suite.runner,
            config: engine.suite.config,
            driveResidualModel: engine.suite.driveResidualModel,
            context: context
        )

        let suiteReport = try engine.run()
        let conformanceReport = try ConformanceReportBuilder.build(
            implementationId: implementationId,
            version: version,
            badges: badges,
            report: suiteReport
        )

        return (suiteReport, conformanceReport)
    }
}

