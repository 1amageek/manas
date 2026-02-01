import Foundation

public struct RampFamilyConfig: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositiveDuration
        case nonPositiveDelta
    }

    public let startValue: Double
    public let slope: Double
    public let duration: TimeInterval
    public let deltaTime: TimeInterval

    public init(
        startValue: Double,
        slope: Double,
        duration: TimeInterval,
        deltaTime: TimeInterval
    ) throws {
        guard startValue.isFinite else { throw ValidationError.nonFinite("startValue") }
        guard slope.isFinite else { throw ValidationError.nonFinite("slope") }
        guard duration.isFinite else { throw ValidationError.nonFinite("duration") }
        guard deltaTime.isFinite else { throw ValidationError.nonFinite("deltaTime") }
        guard duration > 0 else { throw ValidationError.nonPositiveDuration }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.startValue = startValue
        self.slope = slope
        self.duration = duration
        self.deltaTime = deltaTime
    }
}

