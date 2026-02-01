import Foundation

public struct StepFamilyConfig: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositiveDuration
        case nonPositiveDelta
    }

    public let initialValue: Double
    public let stepValue: Double
    public let stepTime: TimeInterval
    public let duration: TimeInterval
    public let deltaTime: TimeInterval

    public init(
        initialValue: Double,
        stepValue: Double,
        stepTime: TimeInterval,
        duration: TimeInterval,
        deltaTime: TimeInterval
    ) throws {
        guard initialValue.isFinite else { throw ValidationError.nonFinite("initialValue") }
        guard stepValue.isFinite else { throw ValidationError.nonFinite("stepValue") }
        guard stepTime.isFinite else { throw ValidationError.nonFinite("stepTime") }
        guard duration.isFinite else { throw ValidationError.nonFinite("duration") }
        guard deltaTime.isFinite else { throw ValidationError.nonFinite("deltaTime") }
        guard duration > 0 else { throw ValidationError.nonPositiveDuration }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.initialValue = initialValue
        self.stepValue = stepValue
        self.stepTime = stepTime
        self.duration = duration
        self.deltaTime = deltaTime
    }
}

