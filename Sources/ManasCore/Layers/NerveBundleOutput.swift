public struct NerveBundleOutput: Sendable, Equatable {
    public let features: [Double]
    public let fastTaps: [Double]
    public let quality: [Double]

    public init(features: [Double], fastTaps: [Double], quality: [Double]) {
        self.features = features
        self.fastTaps = fastTaps
        self.quality = quality
    }
}
