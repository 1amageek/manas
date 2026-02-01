public struct ConformanceSuite<Target: ManasConformanceTarget> {
    public var runner: ConformanceRunner<Target>
    public let config: ConformanceSuiteConfig
    public let driveResidualModel: DriveResidualModel?
    public let context: ConformanceSuiteContext?

    public init(
        runner: ConformanceRunner<Target>,
        config: ConformanceSuiteConfig,
        driveResidualModel: DriveResidualModel? = nil,
        context: ConformanceSuiteContext? = nil
    ) {
        self.runner = runner
        self.config = config
        self.driveResidualModel = driveResidualModel
        self.context = context
    }

    public mutating func run(
        continuityPairs: [ConformanceInputPair],
        totalVariationFamilies: [InputFamily],
        snappingFamilies: [InputFamily],
        phaseVarianceFamilies: [InputFamily],
        modeInductionFamilies: [[InputFamily]]
    ) throws -> ConformanceSuiteReport {
        var continuityReports: [ConformanceSuiteReport.ContinuityReport] = []
        for var pair in continuityPairs {
            let baseRun = try runner.run(family: &pair.base)
            let perturbedRun = try runner.run(family: &pair.perturbed)
            let results = try runner.continuity(
                base: baseRun,
                perturbed: perturbedRun,
                l2: config.l2,
                lInf: config.lInf
            )
            let passes = results.allSatisfy { $0.passes }
            continuityReports.append(
                ConformanceSuiteReport.ContinuityReport(results: results, passes: passes)
            )
        }

        var tvReports: [TotalVariationCheck.Result] = []
        for var family in totalVariationFamilies {
            let run = try runner.run(family: &family)
            let report = try runner.totalVariation(run: run, limit: config.totalVariationLimit)
            tvReports.append(report)
        }

        var snappingReports: [OutputSnappingCheck.Result] = []
        for var family in snappingFamilies {
            let run = try runner.run(family: &family)
            let normalizedSeries: [[Double]]
            if let residualModel = driveResidualModel {
                let residuals = try residualModel.residuals(
                    series: run.outputs,
                    deltaTime: family.deltaTime
                )
                normalizedSeries = try normalizeResiduals(residuals, indices: residualModel.indices)
            } else {
                normalizedSeries = try run.outputs.map { drives in
                    try runner.normalization.normalizedOutput(drives: drives)
                }
            }
            let result = OutputSnappingCheck.evaluate(
                series: normalizedSeries,
                epsilon: config.snappingEpsilon,
                maxClusters: config.snappingMaxClusters
            )
            snappingReports.append(result)
        }

        var phaseVarianceReports: [PhaseVarianceCheck.Result] = []
        var phaseBandwidthReports: [PhaseBandwidthCheck.Result] = []
        var phaseSnappingReports: [PhaseSnappingCheck.Result] = []
        for var family in phaseVarianceFamilies {
            let run = try runner.run(family: &family)
            var phaseSeries: [PhaseIndex: [Double]] = [:]
            for frame in run.inputs {
                for phase in frame.phases {
                    phaseSeries[phase.index, default: []].append(phase.value)
                }
            }
            let result = PhaseVarianceCheck.evaluate(
                phases: phaseSeries,
                minimumVariance: config.minimumPhaseVariance
            )
            phaseVarianceReports.append(result)

            let bandwidthResult = try PhaseBandwidthCheck.evaluate(
                phases: phaseSeries,
                deltaTime: family.deltaTime,
                bandwidthHz: config.phaseBandwidthHz
            )
            phaseBandwidthReports.append(bandwidthResult)

            let snappingResult = PhaseSnappingCheck.evaluate(
                phases: phaseSeries,
                epsilon: config.phaseSnappingEpsilon,
                maxClusters: config.phaseSnappingMaxClusters
            )
            phaseSnappingReports.append(snappingResult)
        }

        var modeInductionReports: [ModeInductionCheck.Result] = []
        for group in modeInductionFamilies {
            var steadyStates: [[Double]] = []
            for var family in group {
                let run = try runner.run(family: &family)
                let steady = try steadyStateVector(
                    outputs: run.outputs,
                    window: config.steadyWindowSize
                )
                steadyStates.append(steady)
            }
            let modeResult = ModeInductionCheck.evaluate(
                steadyStates: steadyStates,
                epsilon: config.modeInductionEpsilon,
                maxModes: config.modeInductionMaxModes
            )
            modeInductionReports.append(modeResult)
        }

        return ConformanceSuiteReport(
            continuity: continuityReports,
            totalVariation: tvReports,
            snapping: snappingReports,
            phaseVariance: phaseVarianceReports,
            phaseBandwidth: phaseBandwidthReports,
            phaseSnapping: phaseSnappingReports,
            modeInduction: modeInductionReports,
            context: context
        )
    }

    public mutating func run(plan: ConformancePlan) throws -> ConformanceSuiteReport {
        try run(
            continuityPairs: plan.continuityPairs,
            totalVariationFamilies: plan.totalVariationFamilies,
            snappingFamilies: plan.snappingFamilies,
            phaseVarianceFamilies: plan.phaseFamilies,
            modeInductionFamilies: plan.modeInductionFamilies
        )
    }

    private func normalizeResiduals(
        _ residuals: [[Double]],
        indices: [DriveIndex]
    ) throws -> [[Double]] {
        var normalized: [[Double]] = []
        normalized.reserveCapacity(residuals.count)

        for row in residuals {
            var values: [DriveIndex: Double] = [:]
            for (index, value) in zip(indices, row) {
                values[index] = value
            }
            normalized.append(try runner.normalization.drive.normalizedVector(values: values))
        }

        return normalized
    }

    private func steadyStateVector(
        outputs: [[DriveIntent]],
        window: Int
    ) throws -> [Double] {
        let count = outputs.count
        let start = max(0, count - window)
        guard start < count else { return [] }

        var sum: [Double] = []
        var samples: Int = 0

        for idx in start..<count {
            let normalized = try runner.normalization.normalizedOutput(drives: outputs[idx])
            if sum.isEmpty {
                sum = Array(repeating: 0.0, count: normalized.count)
            }
            for j in 0..<normalized.count {
                sum[j] += normalized[j]
            }
            samples += 1
        }

        guard samples > 0 else { return [] }
        return sum.map { $0 / Double(samples) }
    }
}
