public struct ManasParameters: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case missingThreshold(PerceptionIndex)
        case missingWeight(PerceptionIndex)
    }

    public let weights: EnergyWeights
    public let thresholds: [PerceptionIndex: PerceptionThresholds]
    public let globalThresholds: GlobalThresholds
    public let driveLimits: DriveLimits
    public let updatePeriod: UpdateRates

    public init(
        weights: EnergyWeights,
        thresholds: [PerceptionIndex: PerceptionThresholds],
        globalThresholds: GlobalThresholds,
        driveLimits: DriveLimits,
        updatePeriod: UpdateRates
    ) throws {
        for index in weights.indices {
            guard thresholds[index] != nil else {
                throw ValidationError.missingThreshold(index)
            }
        }

        for (index, _) in thresholds {
            guard weights.weight(for: index) != nil else {
                throw ValidationError.missingWeight(index)
            }
        }

        self.weights = weights
        self.thresholds = thresholds
        self.globalThresholds = globalThresholds
        self.driveLimits = driveLimits
        self.updatePeriod = updatePeriod
    }
}

