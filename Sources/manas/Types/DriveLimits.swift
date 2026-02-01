public struct DriveLimits: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case emptyRange(DriveIndex)
        case nonFiniteRange(DriveIndex)
        case missingLimit(DriveIndex)
        case outOfBounds(DriveIndex, Double, ClosedRange<Double>)
    }

    private let limits: [DriveIndex: ClosedRange<Double>]

    public init(limits: [DriveIndex: ClosedRange<Double>]) throws {
        for (index, range) in limits {
            guard range.lowerBound.isFinite, range.upperBound.isFinite else {
                throw ValidationError.nonFiniteRange(index)
            }
            guard range.lowerBound <= range.upperBound else {
                throw ValidationError.emptyRange(index)
            }
        }

        self.limits = limits
    }

    public func range(for index: DriveIndex) -> ClosedRange<Double>? {
        limits[index]
    }

    public var indices: [DriveIndex] {
        limits.keys.sorted { $0.rawValue < $1.rawValue }
    }

    public func validate(_ intent: DriveIntent) throws {
        guard let range = limits[intent.index] else {
            throw ValidationError.missingLimit(intent.index)
        }
        guard range.contains(intent.activation) else {
            throw ValidationError.outOfBounds(intent.index, intent.activation, range)
        }
    }

    public func clamped(_ intent: DriveIntent) throws -> DriveIntent {
        guard let range = limits[intent.index] else {
            throw ValidationError.missingLimit(intent.index)
        }

        let clampedValue = min(max(intent.activation, range.lowerBound), range.upperBound)
        return try DriveIntent(index: intent.index, activation: clampedValue)
    }
}
