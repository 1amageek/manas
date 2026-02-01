import Foundation

public struct ActuatorBounds: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
    }

    public let saturationLimit: Double
    public let rateLimit: Double
    public let delay: TimeInterval

    public init(
        saturationLimit: Double,
        rateLimit: Double,
        delay: TimeInterval
    ) throws {
        try ActuatorBounds.validateNonNegativeFinite(saturationLimit, field: "saturationLimit")
        try ActuatorBounds.validateNonNegativeFinite(rateLimit, field: "rateLimit")
        try ActuatorBounds.validateNonNegativeFinite(delay, field: "delay")

        self.saturationLimit = saturationLimit
        self.rateLimit = rateLimit
        self.delay = delay
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }
}

