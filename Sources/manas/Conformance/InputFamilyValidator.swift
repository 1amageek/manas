public struct InputFamilyValidator {
    public enum ValidationError: Error, Equatable {
        case nonFiniteEnergy(PerceptionIndex)
        case negativeEnergy(PerceptionIndex)
        case energyOutOfRange(PerceptionIndex, Double, Double)
        case nonFinitePhase(PhaseIndex)
        case phaseOutOfRange(PhaseIndex, Double, Double)
    }

    public static func validate(
        family: InputFamily,
        energyRanges: NormalizationMap<PerceptionIndex>,
        phaseRanges: NormalizationMap<PhaseIndex>
    ) throws {
        var copy = family
        let frames = try copy.frames()

        for frame in frames {
            for energy in frame.energies {
                guard energy.value.isFinite else {
                    throw ValidationError.nonFiniteEnergy(energy.index)
                }
                guard energy.value >= 0 else {
                    throw ValidationError.negativeEnergy(energy.index)
                }
                let maxValue = try energyRanges.range(for: energy.index)
                if energy.value > maxValue {
                    throw ValidationError.energyOutOfRange(energy.index, energy.value, maxValue)
                }
            }
            for phase in frame.phases {
                guard phase.value.isFinite else {
                    throw ValidationError.nonFinitePhase(phase.index)
                }
                let maxValue = try phaseRanges.range(for: phase.index)
                if abs(phase.value) > maxValue {
                    throw ValidationError.phaseOutOfRange(phase.index, phase.value, maxValue)
                }
            }
        }
    }
}
