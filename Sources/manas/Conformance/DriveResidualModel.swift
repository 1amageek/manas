import Foundation

public struct DriveResidualModel: Sendable {
    public enum ValidationError: Error, Equatable {
        case nonFiniteDelta
        case nonPositiveDelta
        case missingLimit(DriveIndex)
    }

    private let driveLimits: DriveLimits
    private let rateLimits: DriveRateLimits?

    public init(
        driveLimits: DriveLimits,
        rateLimits: DriveRateLimits? = nil
    ) {
        self.driveLimits = driveLimits
        self.rateLimits = rateLimits
    }

    public var indices: [DriveIndex] {
        driveLimits.indices
    }

    public func residuals(
        series: [[DriveIntent]],
        deltaTime: TimeInterval
    ) throws -> [[Double]] {
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        let indices = driveLimits.indices
        var lastValues: [DriveIndex: Double] = [:]
        var residualSeries: [[Double]] = []
        residualSeries.reserveCapacity(series.count)

        for drives in series {
            var driveMap: [DriveIndex: Double] = [:]
            for drive in drives {
                driveMap[drive.index] = drive.activation
            }

            var residuals: [Double] = []
            residuals.reserveCapacity(indices.count)

            for index in indices {
                guard let range = driveLimits.range(for: index) else {
                    throw ValidationError.missingLimit(index)
                }
                let rawValue = driveMap[index] ?? 0.0
                let clamped = min(max(rawValue, range.lowerBound), range.upperBound)
                let limited = applyRateLimit(
                    index: index,
                    value: clamped,
                    deltaTime: deltaTime,
                    lastValues: &lastValues
                )
                residuals.append(rawValue - limited)
            }

            residualSeries.append(residuals)
        }

        return residualSeries
    }

    private func applyRateLimit(
        index: DriveIndex,
        value: Double,
        deltaTime: TimeInterval,
        lastValues: inout [DriveIndex: Double]
    ) -> Double {
        guard let maxRate = rateLimits?.maxRate(for: index) else {
            lastValues[index] = value
            return value
        }

        let previous = lastValues[index] ?? value
        let maxDelta = maxRate * deltaTime
        let delta = value - previous
        let limitedDelta = min(max(delta, -maxDelta), maxDelta)
        let output = previous + limitedDelta
        lastValues[index] = output
        return output
    }
}
