import Foundation

public struct SignalSample<Index: OrderedIndex>: Sendable, Equatable {
    public let index: Index
    public let value: Double
    public let time: TimeInterval

    public init(index: Index, value: Double, time: TimeInterval) {
        self.index = index
        self.value = value
        self.time = time
    }
}

