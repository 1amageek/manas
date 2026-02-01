public struct LearningReport: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
    }

    public let parameterDeltaNorm: Double
    public let parameterDerivativeNorm: Double

    public init(parameterDeltaNorm: Double, parameterDerivativeNorm: Double) throws {
        try LearningReport.validateNonNegativeFinite(parameterDeltaNorm, field: "parameterDeltaNorm")
        try LearningReport.validateNonNegativeFinite(parameterDerivativeNorm, field: "parameterDerivativeNorm")

        self.parameterDeltaNorm = parameterDeltaNorm
        self.parameterDerivativeNorm = parameterDerivativeNorm
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }
}

