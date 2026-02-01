import Foundation

public struct InputFrame: Sendable, Equatable {
    public let time: TimeInterval
    public let energies: [SignalSample<PerceptionIndex>]
    public let phases: [SignalSample<PhaseIndex>]

    public init(
        time: TimeInterval,
        energies: [SignalSample<PerceptionIndex>],
        phases: [SignalSample<PhaseIndex>]
    ) {
        self.time = time
        self.energies = energies
        self.phases = phases
    }
}

