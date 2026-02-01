import Foundation

public struct LearningConstraints: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositive(String)
        case negative(String)
    }

    public let minUpdatePeriod: TimeInterval
    public let maxParameterDeltaNorm: Double
    public let maxParameterDerivativeNorm: Double

    public init(
        minUpdatePeriod: TimeInterval,
        maxParameterDeltaNorm: Double,
        maxParameterDerivativeNorm: Double
    ) throws {
        try LearningConstraints.validatePositiveFinite(minUpdatePeriod, field: "minUpdatePeriod")
        try LearningConstraints.validateNonNegativeFinite(maxParameterDeltaNorm, field: "maxParameterDeltaNorm")
        try LearningConstraints.validateNonNegativeFinite(maxParameterDerivativeNorm, field: "maxParameterDerivativeNorm")

        self.minUpdatePeriod = minUpdatePeriod
        self.maxParameterDeltaNorm = maxParameterDeltaNorm
        self.maxParameterDerivativeNorm = maxParameterDerivativeNorm
    }

    private static func validatePositiveFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value > 0 else { throw ValidationError.nonPositive(field) }
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }
}

