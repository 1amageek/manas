import Foundation

public struct PRBSFamilyConfig: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositiveDuration
        case nonPositiveDelta
        case nonPositiveSwitchPeriod
    }

    public let amplitude: Double
    public let switchPeriod: TimeInterval
    public let cutoffHz: Double
    public let seed: UInt64
    public let duration: TimeInterval
    public let deltaTime: TimeInterval

    public init(
        amplitude: Double,
        switchPeriod: TimeInterval,
        cutoffHz: Double,
        seed: UInt64,
        duration: TimeInterval,
        deltaTime: TimeInterval
    ) throws {
        guard amplitude.isFinite else { throw ValidationError.nonFinite("amplitude") }
        guard switchPeriod.isFinite else { throw ValidationError.nonFinite("switchPeriod") }
        guard cutoffHz.isFinite else { throw ValidationError.nonFinite("cutoffHz") }
        guard duration.isFinite else { throw ValidationError.nonFinite("duration") }
        guard deltaTime.isFinite else { throw ValidationError.nonFinite("deltaTime") }
        guard switchPeriod > 0 else { throw ValidationError.nonPositiveSwitchPeriod }
        guard duration > 0 else { throw ValidationError.nonPositiveDuration }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.amplitude = amplitude
        self.switchPeriod = switchPeriod
        self.cutoffHz = cutoffHz
        self.seed = seed
        self.duration = duration
        self.deltaTime = deltaTime
    }
}

