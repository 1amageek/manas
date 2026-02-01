import Foundation

public struct PRBSSignal: SignalGenerator {
    public enum ValidationError: Error, Equatable {
        case nonFiniteAmplitude
        case nonFinitePeriod
        case nonPositivePeriod
    }

    public let amplitude: Double
    public let switchPeriod: TimeInterval
    private var generator: SplitMix64
    private var currentValue: Double
    private var nextSwitchTime: TimeInterval

    public init(
        amplitude: Double,
        switchPeriod: TimeInterval,
        seed: UInt64
    ) throws {
        guard amplitude.isFinite else { throw ValidationError.nonFiniteAmplitude }
        guard switchPeriod.isFinite else { throw ValidationError.nonFinitePeriod }
        guard switchPeriod > 0 else { throw ValidationError.nonPositivePeriod }

        self.amplitude = amplitude
        self.switchPeriod = switchPeriod
        self.generator = SplitMix64(seed: seed)
        self.currentValue = amplitude
        self.nextSwitchTime = 0.0
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        if time >= nextSwitchTime {
            let bit = generator.next() & 1
            currentValue = bit == 0 ? -amplitude : amplitude
            nextSwitchTime = time + switchPeriod
        }
        return currentValue
    }
}

