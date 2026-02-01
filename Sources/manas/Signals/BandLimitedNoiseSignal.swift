import Foundation

public struct BandLimitedNoiseSignal: SignalGenerator {
    public enum ValidationError: Error, Equatable {
        case nonFiniteAmplitude
        case nonFiniteDelta
        case nonPositiveDelta
    }

    public let amplitude: Double
    public let deltaTime: TimeInterval
    private var generator: SplitMix64
    private var filter: LowPassFilter

    public init(
        amplitude: Double,
        cutoffHz: Double,
        deltaTime: TimeInterval,
        seed: UInt64
    ) throws {
        guard amplitude.isFinite else { throw ValidationError.nonFiniteAmplitude }
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.amplitude = amplitude
        self.deltaTime = deltaTime
        self.generator = SplitMix64(seed: seed)
        self.filter = try LowPassFilter(cutoffHz: cutoffHz, initialValue: 0.0)
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        _ = time
        let raw = (generator.nextDouble() * 2.0 - 1.0) * amplitude
        return try filter.apply(input: raw, deltaTime: deltaTime)
    }
}

