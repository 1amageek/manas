public struct NormalizationBundle: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case duplicateIndex
    }

    public let energy: NormalizationMap<PerceptionIndex>
    public let phase: NormalizationMap<PhaseIndex>
    public let drive: NormalizationMap<DriveIndex>

    public init(
        energy: NormalizationMap<PerceptionIndex>,
        phase: NormalizationMap<PhaseIndex>,
        drive: NormalizationMap<DriveIndex>
    ) {
        self.energy = energy
        self.phase = phase
        self.drive = drive
    }

    public func normalizedInput(
        energies: [EnergyState],
        phases: [PhaseState]
    ) throws -> [Double] {
        var energyValues: [PerceptionIndex: Double] = [:]
        for energy in energies {
            energyValues[energy.index] = energy.value
        }

        var phaseValues: [PhaseIndex: Double] = [:]
        for phase in phases {
            phaseValues[phase.index] = phase.value
        }

        let energyVector = try energy.normalizedVector(values: energyValues)
        let phaseVector = try phase.normalizedVector(values: phaseValues)
        return energyVector + phaseVector
    }

    public func normalizedOutput(drives: [DriveIntent]) throws -> [Double] {
        var driveValues: [DriveIndex: Double] = [:]
        for drive in drives {
            driveValues[drive.index] = drive.activation
        }
        return try drive.normalizedVector(values: driveValues)
    }

    public var energyIndices: [PerceptionIndex] {
        energy.indices
    }

    public var phaseIndices: [PhaseIndex] {
        phase.indices
    }
}
