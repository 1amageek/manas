public struct PhaseSnappingCheck: Sendable {
    public struct Result: Sendable, Equatable {
        public let clustersByIndex: [PhaseIndex: Int]
        public let passes: Bool
    }

    public static func evaluate(
        phases: [PhaseIndex: [Double]],
        epsilon: Double,
        maxClusters: Int
    ) -> Result {
        var clustersByIndex: [PhaseIndex: Int] = [:]
        var passes = true

        for (index, series) in phases {
            var centers: [Double] = []
            for value in series {
                if let centerIndex = centers.firstIndex(where: { abs($0 - value) <= epsilon }) {
                    centers[centerIndex] = (centers[centerIndex] + value) / 2.0
                } else {
                    centers.append(value)
                }
            }
            clustersByIndex[index] = centers.count
            if centers.count <= maxClusters {
                passes = false
            }
        }

        return Result(clustersByIndex: clustersByIndex, passes: passes)
    }
}

