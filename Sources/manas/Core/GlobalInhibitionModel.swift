public protocol GlobalInhibitionModel: Sendable {
    func factor(totalEnergy: Double, existThreshold: Double) throws -> Double
}

