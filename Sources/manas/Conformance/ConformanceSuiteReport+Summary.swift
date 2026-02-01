public extension ConformanceSuiteReport {
    struct Summary: Sendable, Codable, Equatable {
        public let continuityPasses: Bool
        public let totalVariationPasses: Bool
        public let snappingPasses: Bool
        public let phaseVariancePasses: Bool
        public let phaseBandwidthPasses: Bool
        public let phaseSnappingPasses: Bool
        public let modeInductionPasses: Bool
        public let passes: Bool
    }

    var summary: Summary {
        let continuityPasses = continuity.allSatisfy { $0.passes }
        let totalVariationPasses = totalVariation.allSatisfy { $0.passes }
        let snappingPasses = snapping.allSatisfy { $0.passes }
        let phaseVariancePasses = phaseVariance.allSatisfy { $0.passes }
        let phaseBandwidthPasses = phaseBandwidth.allSatisfy { $0.passes }
        let phaseSnappingPasses = phaseSnapping.allSatisfy { $0.passes }
        let modeInductionPasses = modeInduction.allSatisfy { $0.passes }

        let passes = continuityPasses
            && totalVariationPasses
            && snappingPasses
            && phaseVariancePasses
            && phaseBandwidthPasses
            && phaseSnappingPasses
            && modeInductionPasses

        return Summary(
            continuityPasses: continuityPasses,
            totalVariationPasses: totalVariationPasses,
            snappingPasses: snappingPasses,
            phaseVariancePasses: phaseVariancePasses,
            phaseBandwidthPasses: phaseBandwidthPasses,
            phaseSnappingPasses: phaseSnappingPasses,
            modeInductionPasses: modeInductionPasses,
            passes: passes
        )
    }
}
