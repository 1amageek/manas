public struct PhaseVarianceCheck: Sendable {
    public struct Result: Sendable, Equatable {
        public let varianceByIndex: [PhaseIndex: Double]
        public let passes: Bool
    }

    public static func evaluate(
        phases: [PhaseIndex: [Double]],
        minimumVariance: Double
    ) -> Result {
        var varianceByIndex: [PhaseIndex: Double] = [:]
        var passes = true

        for (index, series) in phases {
            let variance = computeVariance(series)
            varianceByIndex[index] = variance
            if variance < minimumVariance {
                passes = false
            }
        }

        return Result(varianceByIndex: varianceByIndex, passes: passes)
    }

    private static func computeVariance(_ series: [Double]) -> Double {
        guard !series.isEmpty else { return 0.0 }
        let mean = series.reduce(0.0, +) / Double(series.count)
        let sum = series.reduce(0.0) { $0 + ( $1 - mean ) * ( $1 - mean ) }
        return sum / Double(series.count)
    }
}

