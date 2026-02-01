public struct PerceptionThresholds: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
        case reflexBelowLocal
    }

    public let localEnergy: Double
    public let criticalGradient: Double
    public let reflexEnergy: Double
    public let reflexGradient: Double

    public init(
        localEnergy: Double,
        criticalGradient: Double,
        reflexEnergy: Double,
        reflexGradient: Double
    ) throws {
        try PerceptionThresholds.validateNonNegativeFinite(localEnergy, field: "localEnergy")
        try PerceptionThresholds.validateNonNegativeFinite(criticalGradient, field: "criticalGradient")
        try PerceptionThresholds.validateNonNegativeFinite(reflexEnergy, field: "reflexEnergy")
        try PerceptionThresholds.validateNonNegativeFinite(reflexGradient, field: "reflexGradient")

        guard reflexEnergy >= localEnergy else {
            throw ValidationError.reflexBelowLocal
        }

        self.localEnergy = localEnergy
        self.criticalGradient = criticalGradient
        self.reflexEnergy = reflexEnergy
        self.reflexGradient = reflexGradient
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }
}

