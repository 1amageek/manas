import Foundation

public struct ChirpFamilyConfig: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositiveDuration
        case nonPositiveDelta
    }

    public let amplitude: Double
    public let initialFrequency: Double
    public let finalFrequency: Double
    public let duration: TimeInterval
    public let deltaTime: TimeInterval

    public init(
        amplitude: Double,
        initialFrequency: Double,
        finalFrequency: Double,
        duration: TimeInterval,
        deltaTime: TimeInterval
    ) throws {
        guard amplitude.isFinite else { throw ValidationError.nonFinite("amplitude") }
        guard initialFrequency.isFinite else { throw ValidationError.nonFinite("initialFrequency") }
        guard finalFrequency.isFinite else { throw ValidationError.nonFinite("finalFrequency") }
        guard duration.isFinite else { throw ValidationError.nonFinite("duration") }
        guard deltaTime.isFinite else { throw ValidationError.nonFinite("deltaTime") }
        guard duration > 0 else { throw ValidationError.nonPositiveDuration }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.amplitude = amplitude
        self.initialFrequency = initialFrequency
        self.finalFrequency = finalFrequency
        self.duration = duration
        self.deltaTime = deltaTime
    }
}

