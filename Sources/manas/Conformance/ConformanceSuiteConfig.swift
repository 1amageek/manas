public struct ConformanceSuiteConfig: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
        case negativeCount(String)
    }

    public let l2: Double
    public let lInf: Double
    public let totalVariationLimit: Double
    public let snappingEpsilon: Double
    public let snappingMaxClusters: Int
    public let minimumPhaseVariance: Double
    public let phaseBandwidthHz: Double
    public let phaseSnappingEpsilon: Double
    public let phaseSnappingMaxClusters: Int
    public let modeInductionEpsilon: Double
    public let modeInductionMaxModes: Int
    public let steadyWindowSize: Int

    public init(
        l2: Double,
        lInf: Double,
        totalVariationLimit: Double,
        snappingEpsilon: Double,
        snappingMaxClusters: Int,
        minimumPhaseVariance: Double,
        phaseBandwidthHz: Double,
        phaseSnappingEpsilon: Double,
        phaseSnappingMaxClusters: Int,
        modeInductionEpsilon: Double,
        modeInductionMaxModes: Int,
        steadyWindowSize: Int
    ) throws {
        try ConformanceSuiteConfig.validateNonNegativeFinite(l2, field: "l2")
        try ConformanceSuiteConfig.validateNonNegativeFinite(lInf, field: "lInf")
        try ConformanceSuiteConfig.validateNonNegativeFinite(totalVariationLimit, field: "totalVariationLimit")
        try ConformanceSuiteConfig.validateNonNegativeFinite(snappingEpsilon, field: "snappingEpsilon")
        try ConformanceSuiteConfig.validateNonNegativeFinite(minimumPhaseVariance, field: "minimumPhaseVariance")
        try ConformanceSuiteConfig.validateNonNegativeFinite(phaseBandwidthHz, field: "phaseBandwidthHz")
        try ConformanceSuiteConfig.validateNonNegativeFinite(phaseSnappingEpsilon, field: "phaseSnappingEpsilon")
        try ConformanceSuiteConfig.validateNonNegativeFinite(modeInductionEpsilon, field: "modeInductionEpsilon")
        try ConformanceSuiteConfig.validateNonNegativeCount(snappingMaxClusters, field: "snappingMaxClusters")
        try ConformanceSuiteConfig.validateNonNegativeCount(phaseSnappingMaxClusters, field: "phaseSnappingMaxClusters")
        try ConformanceSuiteConfig.validateNonNegativeCount(modeInductionMaxModes, field: "modeInductionMaxModes")

        self.l2 = l2
        self.lInf = lInf
        self.totalVariationLimit = totalVariationLimit
        self.snappingEpsilon = snappingEpsilon
        self.snappingMaxClusters = snappingMaxClusters
        self.minimumPhaseVariance = minimumPhaseVariance
        self.phaseBandwidthHz = phaseBandwidthHz
        self.phaseSnappingEpsilon = phaseSnappingEpsilon
        self.phaseSnappingMaxClusters = phaseSnappingMaxClusters
        self.modeInductionEpsilon = modeInductionEpsilon
        self.modeInductionMaxModes = modeInductionMaxModes
        self.steadyWindowSize = max(1, steadyWindowSize)
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }

    private static func validateNonNegativeCount(_ value: Int, field: String) throws {
        guard value >= 0 else { throw ValidationError.negativeCount(field) }
    }
}
