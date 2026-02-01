public protocol ReflexPolicy: Sendable {
    func overrides(
        reflexes: [PerceptionIndex],
        regime: Regime,
        baseDrives: [DriveIntent]
    ) throws -> [DriveIntent]?
}

