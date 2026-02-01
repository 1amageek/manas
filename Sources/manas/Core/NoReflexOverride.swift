public struct NoReflexOverride: ReflexPolicy {
    public init() {}

    public func overrides(
        reflexes: [PerceptionIndex],
        regime: Regime,
        baseDrives: [DriveIntent]
    ) throws -> [DriveIntent]? {
        nil
    }
}

