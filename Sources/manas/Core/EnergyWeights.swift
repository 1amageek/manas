public struct EnergyWeights: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(PerceptionIndex)
        case negative(PerceptionIndex)
        case empty
    }

    private let weights: [PerceptionIndex: Double]

    public init(weights: [PerceptionIndex: Double]) throws {
        guard !weights.isEmpty else {
            throw ValidationError.empty
        }

        for (index, value) in weights {
            guard value.isFinite else { throw ValidationError.nonFinite(index) }
            guard value >= 0 else { throw ValidationError.negative(index) }
        }

        self.weights = weights
    }

    public func weight(for index: PerceptionIndex) -> Double? {
        weights[index]
    }

    public var indices: [PerceptionIndex] {
        weights.keys.sorted { $0.rawValue < $1.rawValue }
    }
}

