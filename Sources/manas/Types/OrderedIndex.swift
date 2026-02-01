public protocol OrderedIndex: Hashable, Sendable, Codable, Comparable {
    var rawValue: UInt32 { get }
}

public extension OrderedIndex {
    static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

