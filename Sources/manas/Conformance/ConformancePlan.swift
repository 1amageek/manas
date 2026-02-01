public struct ConformancePlan {
    public let continuityPairs: [ConformanceInputPair]
    public let totalVariationFamilies: [InputFamily]
    public let snappingFamilies: [InputFamily]
    public let phaseFamilies: [InputFamily]
    public let modeInductionFamilies: [[InputFamily]]

    public init(
        continuityPairs: [ConformanceInputPair],
        totalVariationFamilies: [InputFamily],
        snappingFamilies: [InputFamily],
        phaseFamilies: [InputFamily],
        modeInductionFamilies: [[InputFamily]]
    ) {
        self.continuityPairs = continuityPairs
        self.totalVariationFamilies = totalVariationFamilies
        self.snappingFamilies = snappingFamilies
        self.phaseFamilies = phaseFamilies
        self.modeInductionFamilies = modeInductionFamilies
    }
}

