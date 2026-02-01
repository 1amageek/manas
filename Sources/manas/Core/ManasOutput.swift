public struct ManasOutput: Sendable, Codable, Equatable {
    public let regime: Regime
    public let totalEnergy: Double
    public let globalInhibition: Double
    public let reflexes: [PerceptionIndex]
    public let drives: [DriveIntent]

    public init(
        regime: Regime,
        totalEnergy: Double,
        globalInhibition: Double,
        reflexes: [PerceptionIndex],
        drives: [DriveIntent]
    ) {
        self.regime = regime
        self.totalEnergy = totalEnergy
        self.globalInhibition = globalInhibition
        self.reflexes = reflexes
        self.drives = drives
    }
}

