public struct PhaseState: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case nonFinite
    }

    public let index: PhaseIndex
    public let value: Double

    public init(index: PhaseIndex, value: Double) throws {
        guard value.isFinite else {
            throw ValidationError.nonFinite
        }

        self.index = index
        self.value = value
    }
}

