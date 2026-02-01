import Foundation

public struct FilteredPRBSSignal: SignalGenerator {
    public enum ValidationError: Error, Equatable {
        case nonFiniteCutoff
        case nonFiniteDelta
        case nonPositiveDelta
    }

    private var prbs: PRBSSignal
    private var filter: LowPassFilter
    private let deltaTime: TimeInterval

    public init(
        prbs: PRBSSignal,
        cutoffHz: Double,
        deltaTime: TimeInterval
    ) throws {
        guard cutoffHz.isFinite else { throw ValidationError.nonFiniteCutoff }
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.prbs = prbs
        self.filter = try LowPassFilter(cutoffHz: cutoffHz, initialValue: 0.0)
        self.deltaTime = deltaTime
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        let raw = try prbs.sample(at: time)
        return try filter.apply(input: raw, deltaTime: deltaTime)
    }
}

