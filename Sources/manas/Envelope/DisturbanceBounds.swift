public struct DisturbanceBounds: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
    }

    public let torqueAmplitude: Double
    public let torqueBandwidthHz: Double
    public let forceAmplitude: Double

    public init(
        torqueAmplitude: Double,
        torqueBandwidthHz: Double,
        forceAmplitude: Double
    ) throws {
        try DisturbanceBounds.validateNonNegativeFinite(torqueAmplitude, field: "torqueAmplitude")
        try DisturbanceBounds.validateNonNegativeFinite(torqueBandwidthHz, field: "torqueBandwidthHz")
        try DisturbanceBounds.validateNonNegativeFinite(forceAmplitude, field: "forceAmplitude")

        self.torqueAmplitude = torqueAmplitude
        self.torqueBandwidthHz = torqueBandwidthHz
        self.forceAmplitude = forceAmplitude
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }
}

