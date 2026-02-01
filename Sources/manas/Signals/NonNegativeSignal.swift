import Foundation

public struct NonNegativeSignal: SignalGenerator {
    private var base: AnySignalGenerator

    public init(_ base: AnySignalGenerator) {
        self.base = base
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        let value = try base.sample(at: time)
        return max(0.0, value)
    }
}

