public struct ZeroReflexOverride: ReflexPolicy {
    private let driveIndices: [DriveIndex]

    public init(driveLimits: DriveLimits) {
        self.driveIndices = driveLimits.indices
    }

    public func overrides(
        reflexes: [PerceptionIndex],
        regime: Regime,
        baseDrives: [DriveIntent]
    ) throws -> [DriveIntent]? {
        guard !reflexes.isEmpty else {
            return nil
        }
        return try driveIndices.map { index in
            try DriveIntent(index: index, activation: 0.0)
        }
    }
}

