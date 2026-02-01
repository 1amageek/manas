import Foundation

public struct FiniteDifferenceEnergyGradientEstimator: EnergyGradientEstimator {
    public enum ValidationError: Error, Equatable {
        case nonFiniteDelta
        case nonPositiveDelta
    }

    private var previous: [PerceptionIndex: Double]

    public init() {
        self.previous = [:]
    }

    public mutating func reset() {
        previous = [:]
    }

    public mutating func estimate(
        energies: [EnergyState],
        deltaTime: TimeInterval
    ) throws -> [EnergyGradient] {
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        var gradients: [EnergyGradient] = []
        gradients.reserveCapacity(energies.count)

        for energy in energies {
            let prior = previous[energy.index] ?? energy.value
            let gradientValue = (energy.value - prior) / deltaTime
            gradients.append(try EnergyGradient(index: energy.index, value: gradientValue))
            previous[energy.index] = energy.value
        }

        return gradients
    }
}
