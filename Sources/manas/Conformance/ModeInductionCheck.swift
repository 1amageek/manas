public struct ModeInductionCheck: Sendable {
    public struct Result: Sendable, Equatable {
        public let modeCount: Int
        public let passes: Bool
    }

    public static func evaluate(
        steadyStates: [[Double]],
        epsilon: Double,
        maxModes: Int
    ) -> Result {
        var centers: [[Double]] = []

        for state in steadyStates {
            if let index = centers.firstIndex(where: { vectorsClose($0, state, epsilon: epsilon) }) {
                centers[index] = averageVector(centers[index], state)
            } else {
                centers.append(state)
            }
        }

        let passes = centers.count > maxModes
        return Result(modeCount: centers.count, passes: passes)
    }

    private static func vectorsClose(_ lhs: [Double], _ rhs: [Double], epsilon: Double) -> Bool {
        let count = min(lhs.count, rhs.count)
        for idx in 0..<count {
            if abs(lhs[idx] - rhs[idx]) > epsilon {
                return false
            }
        }
        return true
    }

    private static func averageVector(_ lhs: [Double], _ rhs: [Double]) -> [Double] {
        let count = min(lhs.count, rhs.count)
        var result: [Double] = []
        result.reserveCapacity(count)
        for idx in 0..<count {
            result.append((lhs[idx] + rhs[idx]) / 2.0)
        }
        return result
    }
}

