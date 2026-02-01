import Foundation

public struct ChirpSignal: SignalGenerator {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositiveDuration
    }

    public let amplitude: Double
    public let initialFrequency: Double
    public let finalFrequency: Double
    public let duration: TimeInterval

    public init(
        amplitude: Double,
        initialFrequency: Double,
        finalFrequency: Double,
        duration: TimeInterval
    ) throws {
        guard amplitude.isFinite else { throw ValidationError.nonFinite("amplitude") }
        guard initialFrequency.isFinite else { throw ValidationError.nonFinite("initialFrequency") }
        guard finalFrequency.isFinite else { throw ValidationError.nonFinite("finalFrequency") }
        guard duration.isFinite else { throw ValidationError.nonFinite("duration") }
        guard duration > 0 else { throw ValidationError.nonPositiveDuration }

        self.amplitude = amplitude
        self.initialFrequency = initialFrequency
        self.finalFrequency = finalFrequency
        self.duration = duration
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        let clampedTime = min(max(time, 0.0), duration)
        let k = (finalFrequency - initialFrequency) / duration
        let phase = 2.0 * Double.pi * (initialFrequency * clampedTime + 0.5 * k * clampedTime * clampedTime)
        return amplitude * sin(phase)
    }
}

