public struct ConformanceSuiteReport: Sendable, Equatable {
    public struct ContinuityReport: Sendable, Equatable {
        public let results: [ContinuityCheck.Result]
        public let passes: Bool
    }

    public let continuity: [ContinuityReport]
    public let totalVariation: [TotalVariationCheck.Result]
    public let snapping: [OutputSnappingCheck.Result]
    public let phaseVariance: [PhaseVarianceCheck.Result]
    public let phaseBandwidth: [PhaseBandwidthCheck.Result]
    public let phaseSnapping: [PhaseSnappingCheck.Result]
    public let modeInduction: [ModeInductionCheck.Result]
    public let context: ConformanceSuiteContext?

    public init(
        continuity: [ContinuityReport],
        totalVariation: [TotalVariationCheck.Result],
        snapping: [OutputSnappingCheck.Result],
        phaseVariance: [PhaseVarianceCheck.Result],
        phaseBandwidth: [PhaseBandwidthCheck.Result],
        phaseSnapping: [PhaseSnappingCheck.Result],
        modeInduction: [ModeInductionCheck.Result],
        context: ConformanceSuiteContext? = nil
    ) {
        self.continuity = continuity
        self.totalVariation = totalVariation
        self.snapping = snapping
        self.phaseVariance = phaseVariance
        self.phaseBandwidth = phaseBandwidth
        self.phaseSnapping = phaseSnapping
        self.modeInduction = modeInduction
        self.context = context
    }
}
