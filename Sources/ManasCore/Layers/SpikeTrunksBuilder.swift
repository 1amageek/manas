import Foundation

public struct SpikeTrunksBuilder: TrunkBuilder {
    public struct Configuration: Sendable, Equatable {
        public let spikeGain: Double

        public init(spikeGain: Double) {
            self.spikeGain = spikeGain
        }
    }

    public var configuration: Configuration
    private var lastFastTaps: [Double]

    public init(configuration: Configuration) {
        self.configuration = configuration
        self.lastFastTaps = []
    }

    public mutating func build(from gated: GatedBundle, time: TimeInterval) throws -> TrunkBundle {
        _ = time
        if lastFastTaps.count != gated.fastTaps.count {
            lastFastTaps = Array(repeating: 0.0, count: gated.fastTaps.count)
        }

        let spikes = zip(gated.fastTaps, lastFastTaps).map { current, previous in
            abs(current - previous) * configuration.spikeGain
        }

        lastFastTaps = gated.fastTaps

        let energy = gated.features.map { max(0.0, abs($0)) }
        let phase = gated.features
        let quality = gated.quality

        return TrunkBundle(energy: energy, phase: phase, quality: quality, spike: spikes)
    }
}
