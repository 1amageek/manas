public struct InputFamilyFactory {
    public static func stepFamily(
        config: StepFamilyConfig,
        energyIndices: [PerceptionIndex],
        phaseIndices: [PhaseIndex]
    ) throws -> InputFamily {
        let base = try StepSignal(
            initialValue: config.initialValue,
            stepValue: config.stepValue,
            stepTime: config.stepTime
        )
        let energyChannels = energyIndices.map { index in
            let generator = AnySignalGenerator(NonNegativeSignal(AnySignalGenerator(base)))
            return SignalChannel(index: index, generator: generator)
        }
        let phaseChannels = phaseIndices.map { index in
            let generator = AnySignalGenerator(base)
            return SignalChannel(index: index, generator: generator)
        }

        return try InputFamily(
            duration: config.duration,
            deltaTime: config.deltaTime,
            energyChannels: energyChannels,
            phaseChannels: phaseChannels
        )
    }

    public static func rampFamily(
        config: RampFamilyConfig,
        energyIndices: [PerceptionIndex],
        phaseIndices: [PhaseIndex]
    ) throws -> InputFamily {
        let base = try RampSignal(startValue: config.startValue, slope: config.slope)
        let energyChannels = energyIndices.map { index in
            let generator = AnySignalGenerator(NonNegativeSignal(AnySignalGenerator(base)))
            return SignalChannel(index: index, generator: generator)
        }
        let phaseChannels = phaseIndices.map { index in
            let generator = AnySignalGenerator(base)
            return SignalChannel(index: index, generator: generator)
        }

        return try InputFamily(
            duration: config.duration,
            deltaTime: config.deltaTime,
            energyChannels: energyChannels,
            phaseChannels: phaseChannels
        )
    }

    public static func prbsFamily(
        config: PRBSFamilyConfig,
        energyIndices: [PerceptionIndex],
        phaseIndices: [PhaseIndex]
    ) throws -> InputFamily {
        let prbs = try PRBSSignal(
            amplitude: config.amplitude,
            switchPeriod: config.switchPeriod,
            seed: config.seed
        )
        let filtered = try FilteredPRBSSignal(
            prbs: prbs,
            cutoffHz: config.cutoffHz,
            deltaTime: config.deltaTime
        )

        let energyChannels = energyIndices.map { index in
            let generator = AnySignalGenerator(NonNegativeSignal(AnySignalGenerator(filtered)))
            return SignalChannel(index: index, generator: generator)
        }
        let phaseChannels = phaseIndices.map { index in
            let generator = AnySignalGenerator(filtered)
            return SignalChannel(index: index, generator: generator)
        }

        return try InputFamily(
            duration: config.duration,
            deltaTime: config.deltaTime,
            energyChannels: energyChannels,
            phaseChannels: phaseChannels
        )
    }

    public static func chirpFamily(
        config: ChirpFamilyConfig,
        energyIndices: [PerceptionIndex],
        phaseIndices: [PhaseIndex]
    ) throws -> InputFamily {
        let chirp = try ChirpSignal(
            amplitude: config.amplitude,
            initialFrequency: config.initialFrequency,
            finalFrequency: config.finalFrequency,
            duration: config.duration
        )
        let energyChannels = energyIndices.map { index in
            let generator = AnySignalGenerator(NonNegativeSignal(AnySignalGenerator(chirp)))
            return SignalChannel(index: index, generator: generator)
        }
        let phaseChannels = phaseIndices.map { index in
            let generator = AnySignalGenerator(chirp)
            return SignalChannel(index: index, generator: generator)
        }

        return try InputFamily(
            duration: config.duration,
            deltaTime: config.deltaTime,
            energyChannels: energyChannels,
            phaseChannels: phaseChannels
        )
    }

    public static func noiseFamily(
        config: NoiseFamilyConfig,
        energyIndices: [PerceptionIndex],
        phaseIndices: [PhaseIndex]
    ) throws -> InputFamily {
        let noise = try BandLimitedNoiseSignal(
            amplitude: config.amplitude,
            cutoffHz: config.cutoffHz,
            deltaTime: config.deltaTime,
            seed: config.seed
        )
        let energyChannels = energyIndices.map { index in
            let generator = AnySignalGenerator(NonNegativeSignal(AnySignalGenerator(noise)))
            return SignalChannel(index: index, generator: generator)
        }
        let phaseChannels = phaseIndices.map { index in
            let generator = AnySignalGenerator(noise)
            return SignalChannel(index: index, generator: generator)
        }

        return try InputFamily(
            duration: config.duration,
            deltaTime: config.deltaTime,
            energyChannels: energyChannels,
            phaseChannels: phaseChannels
        )
    }
}
