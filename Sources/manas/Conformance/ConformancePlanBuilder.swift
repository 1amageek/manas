import Foundation

public struct ConformancePlanBuilder {
    public enum ValidationError: Error, Equatable {
        case emptyIndices
    }

    public static func build(
        coverage: ConformanceCoverageConfig,
        energyIndices: [PerceptionIndex],
        phaseIndices: [PhaseIndex]
    ) throws -> ConformancePlan {
        guard !energyIndices.isEmpty || !phaseIndices.isEmpty else {
            throw ValidationError.emptyIndices
        }

        let amplitudeValues = coverage.amplitude.values()
        let slopeValues = coverage.slope.values()
        let frequencyValues = coverage.frequency.values()
        let frequencyBands = coverage.frequency.bands

        var continuityPairs: [ConformanceInputPair] = []
        for amplitude in amplitudeValues {
            let baseConfig = try StepFamilyConfig(
                initialValue: 0.0,
                stepValue: amplitude,
                stepTime: coverage.stepTime,
                duration: coverage.duration,
                deltaTime: coverage.deltaTime
            )
            let perturbedConfig = try StepFamilyConfig(
                initialValue: 0.0,
                stepValue: amplitude + coverage.perturbationDelta,
                stepTime: coverage.stepTime,
                duration: coverage.duration,
                deltaTime: coverage.deltaTime
            )
            let baseFamily = try InputFamilyFactory.stepFamily(
                config: baseConfig,
                energyIndices: energyIndices,
                phaseIndices: phaseIndices
            )
            let perturbedFamily = try InputFamilyFactory.stepFamily(
                config: perturbedConfig,
                energyIndices: energyIndices,
                phaseIndices: phaseIndices
            )
            continuityPairs.append(ConformanceInputPair(base: baseFamily, perturbed: perturbedFamily))
        }

        var totalVariationFamilies: [InputFamily] = []
        for slope in slopeValues {
            let rampConfig = try RampFamilyConfig(
                startValue: 0.0,
                slope: slope,
                duration: coverage.duration,
                deltaTime: coverage.deltaTime
            )
            totalVariationFamilies.append(
                try InputFamilyFactory.rampFamily(
                    config: rampConfig,
                    energyIndices: energyIndices,
                    phaseIndices: phaseIndices
                )
            )
        }

        var snappingFamilies: [InputFamily] = []
        var phaseFamilies: [InputFamily] = []
        var modeInductionFamilies: [[InputFamily]] = []

        var seedOffset: UInt64 = 0
        for amplitude in amplitudeValues {
            for frequency in frequencyValues {
                let cutoff = max(frequency, coverage.minimumCutoffHz)
                let switchPeriod = switchPeriodForFrequency(frequency, duration: coverage.duration)

                let prbsConfig = try PRBSFamilyConfig(
                    amplitude: amplitude,
                    switchPeriod: switchPeriod,
                    cutoffHz: cutoff,
                    seed: coverage.seedBase &+ seedOffset,
                    duration: coverage.duration,
                    deltaTime: coverage.deltaTime
                )
                snappingFamilies.append(
                    try InputFamilyFactory.prbsFamily(
                        config: prbsConfig,
                        energyIndices: energyIndices,
                        phaseIndices: phaseIndices
                    )
                )

                let noiseConfig = try NoiseFamilyConfig(
                    amplitude: amplitude,
                    cutoffHz: cutoff,
                    seed: coverage.seedBase &+ seedOffset &+ 1000,
                    duration: coverage.duration,
                    deltaTime: coverage.deltaTime
                )
                let noiseFamily = try InputFamilyFactory.noiseFamily(
                    config: noiseConfig,
                    energyIndices: energyIndices,
                    phaseIndices: phaseIndices
                )
                snappingFamilies.append(noiseFamily)
                phaseFamilies.append(noiseFamily)

                seedOffset &+= 1
            }
        }

        for amplitude in amplitudeValues {
            for band in frequencyBands {
                let chirpConfig = try ChirpFamilyConfig(
                    amplitude: amplitude,
                    initialFrequency: band.minimum,
                    finalFrequency: band.maximum,
                    duration: coverage.duration,
                    deltaTime: coverage.deltaTime
                )
                let chirpFamily = try InputFamilyFactory.chirpFamily(
                    config: chirpConfig,
                    energyIndices: energyIndices,
                    phaseIndices: phaseIndices
                )
                snappingFamilies.append(chirpFamily)
                phaseFamilies.append(chirpFamily)
            }
        }

        for amplitude in amplitudeValues {
            var group: [InputFamily] = []
            for offset in coverage.modeInductionOffsets {
                let config = try StepFamilyConfig(
                    initialValue: 0.0,
                    stepValue: amplitude + offset,
                    stepTime: coverage.stepTime,
                    duration: coverage.duration,
                    deltaTime: coverage.deltaTime
                )
                group.append(
                    try InputFamilyFactory.stepFamily(
                        config: config,
                        energyIndices: energyIndices,
                        phaseIndices: phaseIndices
                    )
                )
            }
            modeInductionFamilies.append(group)
        }

        return ConformancePlan(
            continuityPairs: continuityPairs,
            totalVariationFamilies: totalVariationFamilies,
            snappingFamilies: snappingFamilies,
            phaseFamilies: phaseFamilies,
            modeInductionFamilies: modeInductionFamilies
        )
    }

    private static func switchPeriodForFrequency(
        _ frequency: Double,
        duration: TimeInterval
    ) -> TimeInterval {
        guard frequency > 0 else { return duration }
        return max(1.0 / (2.0 * frequency), 1e-6)
    }
}
