public struct BandCoverage: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case empty
    }

    public let bands: [Band]
    public let strategy: BandSampleStrategy

    public init(bands: [Band], strategy: BandSampleStrategy) throws {
        guard !bands.isEmpty else { throw ValidationError.empty }
        self.bands = bands
        self.strategy = strategy
    }

    public func values() -> [Double] {
        var results: [Double] = []
        for band in bands {
            switch strategy {
            case .midpoint:
                results.append(band.midpoint())
            case .minMidMax:
                let mid = band.midpoint()
                results.append(band.minimum)
                if mid != band.minimum && mid != band.maximum {
                    results.append(mid)
                }
                if band.maximum != band.minimum {
                    results.append(band.maximum)
                }
            }
        }
        return Array(Set(results)).sorted()
    }
}

