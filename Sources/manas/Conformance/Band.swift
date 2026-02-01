public struct Band: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case invalidOrder
    }

    public let minimum: Double
    public let maximum: Double

    public init(minimum: Double, maximum: Double) throws {
        guard minimum.isFinite else { throw ValidationError.nonFinite("minimum") }
        guard maximum.isFinite else { throw ValidationError.nonFinite("maximum") }
        guard minimum <= maximum else { throw ValidationError.invalidOrder }

        self.minimum = minimum
        self.maximum = maximum
    }

    public func midpoint() -> Double {
        (minimum + maximum) / 2.0
    }
}

