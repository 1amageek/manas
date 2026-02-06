import Foundation

public struct BasicTrunksBuilder: TrunkBuilder {
    public init() {}

    public mutating func build(from gated: GatedBundle, time: TimeInterval) throws -> TrunkBundle {
        _ = time
        let energy = gated.features.map { max(0.0, abs($0)) }
        let phase = gated.features
        let quality = gated.quality
        let spike = gated.fastTaps.map { abs($0) }
        return TrunkBundle(energy: energy, phase: phase, quality: quality, spike: spike)
    }
}
