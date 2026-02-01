public struct PhysicalBounds: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
    }

    public let maxAngularRate: Double
    public let maxTiltRadians: Double
    public let maxLinearAcceleration: Double

    public init(
        maxAngularRate: Double,
        maxTiltRadians: Double,
        maxLinearAcceleration: Double
    ) throws {
        try PhysicalBounds.validateNonNegativeFinite(maxAngularRate, field: "maxAngularRate")
        try PhysicalBounds.validateNonNegativeFinite(maxTiltRadians, field: "maxTiltRadians")
        try PhysicalBounds.validateNonNegativeFinite(maxLinearAcceleration, field: "maxLinearAcceleration")

        self.maxAngularRate = maxAngularRate
        self.maxTiltRadians = maxTiltRadians
        self.maxLinearAcceleration = maxLinearAcceleration
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }
}

