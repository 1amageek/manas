import Foundation

public struct NoiseFamilyConfig: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositiveDuration
        case nonPositiveDelta
    }

    public let amplitude: Double
    public let cutoffHz: Double
    public let seed: UInt64
    public let duration: TimeInterval
    public let deltaTime: TimeInterval

    public init(
        amplitude: Double,
        cutoffHz: Double,
        seed: UInt64,
        duration: TimeInterval,
        deltaTime: TimeInterval
    ) throws {
        guard amplitude.isFinite else { throw ValidationError.nonFinite("amplitude") }
        guard cutoffHz.isFinite else { throw ValidationError.nonFinite("cutoffHz") }
        guard duration.isFinite else { throw ValidationError.nonFinite("duration") }
        guard deltaTime.isFinite else { throw ValidationError.nonFinite("deltaTime") }
        guard duration > 0 else { throw ValidationError.nonPositiveDuration }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.amplitude = amplitude
        self.cutoffHz = cutoffHz
        self.seed = seed
        self.duration = duration
        self.deltaTime = deltaTime
    }
}

