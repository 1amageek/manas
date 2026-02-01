public protocol LocalInhibitionModel: Sendable {
    func factor(
        energy: Double,
        gradient: Double,
        thresholds: PerceptionThresholds
    ) throws -> Double
}

