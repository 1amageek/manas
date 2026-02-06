public struct SignalChannel: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case nonFinite
    }

    public let index: UInt32
    public let range: ClosedRange<Double>

    public init(index: UInt32, range: ClosedRange<Double>) throws {
        guard range.lowerBound.isFinite, range.upperBound.isFinite else {
            throw ValidationError.nonFinite
        }
        self.index = index
        self.range = range
    }
}
