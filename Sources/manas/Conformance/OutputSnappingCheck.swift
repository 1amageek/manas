public struct OutputSnappingCheck: Sendable {
    public struct Result: Sendable, Equatable {
        public let clustersPerDrive: [Int]
        public let passes: Bool
    }

    public static func evaluate(
        series: [[Double]],
        epsilon: Double,
        maxClusters: Int
    ) -> Result {
        guard !series.isEmpty else {
            return Result(clustersPerDrive: [], passes: true)
        }

        let driveCount = series.first?.count ?? 0
        var clustersPerDrive: [Int] = Array(repeating: 0, count: driveCount)

        for driveIndex in 0..<driveCount {
            var centers: [Double] = []
            for sample in series {
                guard driveIndex < sample.count else { continue }
                let value = sample[driveIndex]
                if let centerIndex = centers.firstIndex(where: { abs($0 - value) <= epsilon }) {
                    centers[centerIndex] = (centers[centerIndex] + value) / 2.0
                } else {
                    centers.append(value)
                }
            }
            clustersPerDrive[driveIndex] = centers.count
        }

        let passes = clustersPerDrive.allSatisfy { $0 > maxClusters }
        return Result(clustersPerDrive: clustersPerDrive, passes: passes)
    }
}

