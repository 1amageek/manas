public struct EnergyState: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case nonFinite
        case negativeValue
    }

    public let index: PerceptionIndex
    public let value: Double

    public init(index: PerceptionIndex, value: Double) throws {
        guard value.isFinite else {
            throw ValidationError.nonFinite
        }
        guard value >= 0 else {
            throw ValidationError.negativeValue
        }

        self.index = index
        self.value = value
    }
}

