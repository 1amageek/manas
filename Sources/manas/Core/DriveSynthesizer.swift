public protocol DriveSynthesizer: Sendable {
    func synthesize(
        energies: [EnergyState],
        phases: [PhaseState],
        regime: Regime
    ) throws -> [DriveIntent]
}

