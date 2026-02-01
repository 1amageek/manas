import Foundation

public struct ConformanceSuiteEngine<Target: ManasConformanceTarget> {
    public var suite: ConformanceSuite<Target>
    public let coverage: ConformanceCoverageConfig
    public let energyIndices: [PerceptionIndex]
    public let phaseIndices: [PhaseIndex]
    public let updateRates: UpdateRates
    public let deltaTolerance: TimeInterval
    public let validatesInputs: Bool

    public init(
        suite: ConformanceSuite<Target>,
        coverage: ConformanceCoverageConfig,
        energyIndices: [PerceptionIndex],
        phaseIndices: [PhaseIndex],
        updateRates: UpdateRates,
        deltaTolerance: TimeInterval = 0.0,
        validatesInputs: Bool = true
    ) {
        self.suite = suite
        self.coverage = coverage
        self.energyIndices = energyIndices
        self.phaseIndices = phaseIndices
        self.updateRates = updateRates
        self.deltaTolerance = deltaTolerance
        self.validatesInputs = validatesInputs
    }

    public mutating func run() throws -> ConformanceSuiteReport {
        let plan = try ConformancePlanBuilder.build(
            coverage: coverage,
            energyIndices: energyIndices,
            phaseIndices: phaseIndices
        )
        try ConformanceSuiteValidator.validateDeltaTime(
            families: plan.allFamilies,
            updateRates: updateRates,
            tolerance: deltaTolerance
        )
        if validatesInputs {
            for family in plan.allFamilies {
                try InputFamilyValidator.validate(
                    family: family,
                    energyRanges: suite.runner.normalization.energy,
                    phaseRanges: suite.runner.normalization.phase
                )
            }
        }
        return try suite.run(plan: plan)
    }
}
