public struct GatedBundle: Sendable, Equatable {
    public let features: [Double]
    public let fastTaps: [Double]
    public let quality: [Double]
    public let gateFactors: [Double]

    public init(features: [Double], fastTaps: [Double], quality: [Double], gateFactors: [Double]) {
        self.features = features
        self.fastTaps = fastTaps
        self.quality = quality
        self.gateFactors = gateFactors
    }
}
