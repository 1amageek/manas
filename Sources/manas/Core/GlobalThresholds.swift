public struct GlobalThresholds: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite
        case negative
    }

    public let existEnergy: Double

    public init(existEnergy: Double) throws {
        guard existEnergy.isFinite else { throw ValidationError.nonFinite }
        guard existEnergy >= 0 else { throw ValidationError.negative }
        self.existEnergy = existEnergy
    }
}

