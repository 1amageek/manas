public struct EnergyGradient: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite
    }

    public let index: PerceptionIndex
    public let value: Double

    public init(index: PerceptionIndex, value: Double) throws {
        guard value.isFinite else { throw ValidationError.nonFinite }
        self.index = index
        self.value = value
    }
}

