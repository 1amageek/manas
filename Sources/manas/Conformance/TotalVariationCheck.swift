public struct TotalVariationCheck: Sendable {
    public struct Result: Sendable, Equatable {
        public let totalVariation: Double
        public let limit: Double
        public let passes: Bool
    }

    public static func evaluate(series: [[Double]], limit: Double) -> Result {
        let tv = ConformanceMath.totalVariation(series: series)
        let passes = tv <= limit
        return Result(totalVariation: tv, limit: limit, passes: passes)
    }
}

