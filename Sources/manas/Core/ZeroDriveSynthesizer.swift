public struct ZeroDriveSynthesizer: DriveSynthesizer {
    private let driveIndices: [DriveIndex]

    public init(driveLimits: DriveLimits) {
        self.driveIndices = driveLimits.indices
    }

    public func synthesize(
        energies: [EnergyState],
        phases: [PhaseState],
        regime: Regime
    ) throws -> [DriveIntent] {
        try driveIndices.map { index in
            try DriveIntent(index: index, activation: 0.0)
        }
    }
}

