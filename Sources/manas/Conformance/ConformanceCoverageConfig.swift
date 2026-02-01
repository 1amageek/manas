import Foundation

public struct ConformanceCoverageConfig: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositive(String)
        case emptyOffsets
        case negativeCutoff
        case negativeFrequencyBand
    }

    public let duration: TimeInterval
    public let deltaTime: TimeInterval
    public let stepTime: TimeInterval
    public let amplitude: BandCoverage
    public let slope: BandCoverage
    public let frequency: BandCoverage
    public let seedBase: UInt64
    public let perturbationDelta: Double
    public let modeInductionOffsets: [Double]
    public let minimumCutoffHz: Double

    public init(
        duration: TimeInterval,
        deltaTime: TimeInterval,
        stepTime: TimeInterval,
        amplitude: BandCoverage,
        slope: BandCoverage,
        frequency: BandCoverage,
        seedBase: UInt64,
        perturbationDelta: Double,
        modeInductionOffsets: [Double],
        minimumCutoffHz: Double
    ) throws {
        guard duration.isFinite else { throw ValidationError.nonFinite("duration") }
        guard deltaTime.isFinite else { throw ValidationError.nonFinite("deltaTime") }
        guard stepTime.isFinite else { throw ValidationError.nonFinite("stepTime") }
        guard perturbationDelta.isFinite else { throw ValidationError.nonFinite("perturbationDelta") }
        guard minimumCutoffHz.isFinite else { throw ValidationError.nonFinite("minimumCutoffHz") }
        guard duration > 0 else { throw ValidationError.nonPositive("duration") }
        guard deltaTime > 0 else { throw ValidationError.nonPositive("deltaTime") }
        guard minimumCutoffHz >= 0 else { throw ValidationError.negativeCutoff }
        guard !modeInductionOffsets.isEmpty else { throw ValidationError.emptyOffsets }
        if frequency.bands.contains(where: { $0.minimum < 0 }) {
            throw ValidationError.negativeFrequencyBand
        }

        self.duration = duration
        self.deltaTime = deltaTime
        self.stepTime = stepTime
        self.amplitude = amplitude
        self.slope = slope
        self.frequency = frequency
        self.seedBase = seedBase
        self.perturbationDelta = perturbationDelta
        self.modeInductionOffsets = modeInductionOffsets
        self.minimumCutoffHz = minimumCutoffHz
    }
}
