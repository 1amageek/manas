import Foundation

public protocol ManasConformanceTarget: Sendable {
    mutating func reset()
    mutating func step(
        energies: [EnergyState],
        phases: [PhaseState],
        deltaTime: TimeInterval
    ) throws -> [DriveIntent]
}
