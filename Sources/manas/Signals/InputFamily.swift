import Foundation

public struct InputFamily {
    public enum ValidationError: Error, Equatable {
        case nonFiniteDuration
        case nonPositiveDuration
        case nonFiniteDelta
        case nonPositiveDelta
    }

    public let duration: TimeInterval
    public let deltaTime: TimeInterval
    public var energyChannels: [SignalChannel<PerceptionIndex>]
    public var phaseChannels: [SignalChannel<PhaseIndex>]

    public init(
        duration: TimeInterval,
        deltaTime: TimeInterval,
        energyChannels: [SignalChannel<PerceptionIndex>],
        phaseChannels: [SignalChannel<PhaseIndex>]
    ) throws {
        guard duration.isFinite else { throw ValidationError.nonFiniteDuration }
        guard duration > 0 else { throw ValidationError.nonPositiveDuration }
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        self.duration = duration
        self.deltaTime = deltaTime
        self.energyChannels = energyChannels
        self.phaseChannels = phaseChannels
    }

    public mutating func frames() throws -> [InputFrame] {
        let steps = Int((duration / deltaTime).rounded(.down))
        var result: [InputFrame] = []
        result.reserveCapacity(steps + 1)

        for step in 0...steps {
            let time = TimeInterval(step) * deltaTime

            var energies: [SignalSample<PerceptionIndex>] = []
            energies.reserveCapacity(energyChannels.count)
            for idx in energyChannels.indices {
                let sample = try energyChannels[idx].generator.sample(at: time)
                energies.append(SignalSample(index: energyChannels[idx].index, value: sample, time: time))
            }

            var phases: [SignalSample<PhaseIndex>] = []
            phases.reserveCapacity(phaseChannels.count)
            for idx in phaseChannels.indices {
                let sample = try phaseChannels[idx].generator.sample(at: time)
                phases.append(SignalSample(index: phaseChannels[idx].index, value: sample, time: time))
            }

            result.append(InputFrame(time: time, energies: energies, phases: phases))
        }

        return result
    }
}
