public struct ActuatorLimit: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFiniteRange
        case emptyRange
        case nonFiniteRate
        case negativeRate
    }

    public let range: ClosedRange<Double>
    public let maxRate: Double?

    public init(range: ClosedRange<Double>, maxRate: Double?) throws {
        guard range.lowerBound.isFinite, range.upperBound.isFinite else {
            throw ValidationError.nonFiniteRange
        }
        guard range.lowerBound <= range.upperBound else {
            throw ValidationError.emptyRange
        }
        if let maxRateValue = maxRate {
            guard maxRateValue.isFinite else {
                throw ValidationError.nonFiniteRate
            }
            guard maxRateValue >= 0 else {
                throw ValidationError.negativeRate
            }
        }

        self.range = range
        self.maxRate = maxRate
    }
}

