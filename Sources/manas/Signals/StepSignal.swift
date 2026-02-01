import Foundation

public struct StepSignal: SignalGenerator {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
    }

    public let initialValue: Double
    public let stepValue: Double
    public let stepTime: TimeInterval

    public init(
        initialValue: Double,
        stepValue: Double,
        stepTime: TimeInterval
    ) throws {
        guard initialValue.isFinite else { throw ValidationError.nonFinite("initialValue") }
        guard stepValue.isFinite else { throw ValidationError.nonFinite("stepValue") }
        guard stepTime.isFinite else { throw ValidationError.nonFinite("stepTime") }

        self.initialValue = initialValue
        self.stepValue = stepValue
        self.stepTime = stepTime
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        time >= stepTime ? stepValue : initialValue
    }
}

