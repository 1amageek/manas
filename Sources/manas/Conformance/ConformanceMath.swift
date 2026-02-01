public struct ConformanceMath {
    public static func l2Norm(_ values: [Double]) -> Double {
        let sum = values.reduce(0.0) { $0 + $1 * $1 }
        return sum.squareRoot()
    }

    public static func lInfNorm(_ values: [Double]) -> Double {
        values.map { abs($0) }.max() ?? 0.0
    }

    public static func l1Norm(_ values: [Double]) -> Double {
        values.reduce(0.0) { $0 + abs($1) }
    }

    public static func totalVariation(series: [[Double]]) -> Double {
        guard series.count >= 2 else {
            return 0.0
        }

        var total: Double = 0.0
        for index in 1..<series.count {
            let previous = series[index - 1]
            let current = series[index]
            let count = min(previous.count, current.count)
            var delta: [Double] = []
            delta.reserveCapacity(count)
            for i in 0..<count {
                delta.append(current[i] - previous[i])
            }
            total += l1Norm(delta)
        }
        return total
    }
}

