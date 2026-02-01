import Foundation

public struct ConformanceTemplate {
    public static func baselineCoverage(
        updateRates: UpdateRates,
        stepTime: TimeInterval,
        amplitudeBands: [Band],
        slopeBands: [Band],
        frequencyBands: [Band],
        seedBase: UInt64,
        perturbationDelta: Double,
        modeInductionOffsets: [Double],
        minimumCutoffHz: Double,
        durationMultiplier: Double = 10.0
    ) throws -> ConformanceCoverageConfig {
        let duration = updateRates.controllerUpdate * durationMultiplier
        return try baselineCoverage(
            duration: duration,
            deltaTime: updateRates.controllerUpdate,
            stepTime: stepTime,
            amplitudeBands: amplitudeBands,
            slopeBands: slopeBands,
            frequencyBands: frequencyBands,
            seedBase: seedBase,
            perturbationDelta: perturbationDelta,
            modeInductionOffsets: modeInductionOffsets,
            minimumCutoffHz: minimumCutoffHz
        )
    }

    public static func baselineCoverage(
        duration: TimeInterval,
        deltaTime: TimeInterval,
        stepTime: TimeInterval,
        amplitudeBands: [Band],
        slopeBands: [Band],
        frequencyBands: [Band],
        seedBase: UInt64,
        perturbationDelta: Double,
        modeInductionOffsets: [Double],
        minimumCutoffHz: Double
    ) throws -> ConformanceCoverageConfig {
        let amplitude = try BandCoverage(bands: amplitudeBands, strategy: .minMidMax)
        let slope = try BandCoverage(bands: slopeBands, strategy: .midpoint)
        let frequency = try BandCoverage(bands: frequencyBands, strategy: .minMidMax)

        return try ConformanceCoverageConfig(
            duration: duration,
            deltaTime: deltaTime,
            stepTime: stepTime,
            amplitude: amplitude,
            slope: slope,
            frequency: frequency,
            seedBase: seedBase,
            perturbationDelta: perturbationDelta,
            modeInductionOffsets: modeInductionOffsets,
            minimumCutoffHz: minimumCutoffHz
        )
    }
}
