import Foundation

public protocol EnergyGradientEstimator: Sendable {
    mutating func reset()
    mutating func estimate(
        energies: [EnergyState],
        deltaTime: TimeInterval
    ) throws -> [EnergyGradient]
}
