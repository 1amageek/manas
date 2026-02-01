public struct DriveRateLimits: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(DriveIndex)
        case negative(DriveIndex)
    }

    private let limits: [DriveIndex: Double]

    public init(limits: [DriveIndex: Double]) throws {
        for (index, value) in limits {
            guard value.isFinite else { throw ValidationError.nonFinite(index) }
            guard value >= 0 else { throw ValidationError.negative(index) }
        }
        self.limits = limits
    }

    public func maxRate(for index: DriveIndex) -> Double? {
        limits[index]
    }
}

