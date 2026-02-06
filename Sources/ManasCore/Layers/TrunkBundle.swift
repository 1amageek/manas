public struct TrunkBundle: Sendable, Equatable {
    public let energy: [Double]
    public let phase: [Double]
    public let quality: [Double]
    public let spike: [Double]

    public init(energy: [Double], phase: [Double], quality: [Double], spike: [Double]) {
        self.energy = energy
        self.phase = phase
        self.quality = quality
        self.spike = spike
    }
}
