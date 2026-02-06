import Foundation

public struct ManasMotorNerve<Mapper: MotorNerveMapper> {
    public enum ValidationError: Error, Equatable {
        case nonFiniteDelta
        case nonPositiveDelta
        case driveCountMismatch(expected: Int, actual: Int)
        case duplicateDriveIndex(DriveIndex)
        case outOfRangeDriveIndex(DriveIndex)
        case parameterCountMismatch(DriveIndex)
        case outOfRangeCorrection(DriveIndex)
    }

    public var mapper: Mapper
    public var primitiveBank: PrimitiveBankDescriptor

    public init(
        mapper: Mapper,
        primitiveBank: PrimitiveBankDescriptor
    ) {
        self.mapper = mapper
        self.primitiveBank = primitiveBank
    }

    public mutating func apply(
        drives: [DriveIntent],
        corrections: [ReflexCorrection],
        telemetry: MotorNerveTelemetry,
        deltaTime: TimeInterval
    ) throws -> [ActuatorValue] {
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        try validate(drives: drives, corrections: corrections)
        let adjusted = try applyCorrections(drives: drives, corrections: corrections)
        return try mapper.map(input: adjusted, telemetry: telemetry)
    }

    private func applyCorrections(
        drives: [DriveIntent],
        corrections: [ReflexCorrection]
    ) throws -> [DriveIntent] {
        guard !corrections.isEmpty else {
            return try drives.map { try bounded($0) }
        }

        var aggregate: [DriveIndex: ReflexAggregate] = [:]
        for correction in corrections {
            var entry = aggregate[correction.driveIndex] ?? ReflexAggregate()
            entry.clampMultiplier *= correction.clampMultiplier
            entry.damping = min(1.0, entry.damping + correction.damping)
            entry.delta += correction.delta
            aggregate[correction.driveIndex] = entry
        }

        return try drives.map { drive in
            guard let entry = aggregate[drive.index] else {
                return try bounded(drive)
            }
            let damped = drive.activation * (1.0 - entry.damping)
            let clamped = damped * entry.clampMultiplier
            let adjusted = clamped + entry.delta
            return try bounded(DriveIntent(index: drive.index, activation: adjusted, parameters: drive.parameters))
        }
    }

    private struct ReflexAggregate {
        var clampMultiplier: Double = 1.0
        var damping: Double = 0.0
        var delta: Double = 0.0
    }

    private func bounded(_ drive: DriveIntent) throws -> DriveIntent {
        let range = try primitiveBank.activationRange(for: drive.index)
        let clamped = min(max(drive.activation, range.lowerBound), range.upperBound)
        return try DriveIntent(index: drive.index, activation: clamped, parameters: drive.parameters)
    }

    private func validate(drives: [DriveIntent], corrections: [ReflexCorrection]) throws {
        let expectedCount = primitiveBank.driveCount
        guard drives.count == expectedCount else {
            throw ValidationError.driveCountMismatch(expected: expectedCount, actual: drives.count)
        }

        var seen: Set<DriveIndex> = []
        for drive in drives {
            guard drive.index.rawValue < UInt32(expectedCount) else {
                throw ValidationError.outOfRangeDriveIndex(drive.index)
            }
            guard seen.insert(drive.index).inserted else {
                throw ValidationError.duplicateDriveIndex(drive.index)
            }
            let descriptor = try primitiveBank.descriptor(for: drive.index)
            if drive.parameters.count != descriptor.parameterCount {
                throw ValidationError.parameterCountMismatch(drive.index)
            }
        }

        for correction in corrections {
            guard correction.driveIndex.rawValue < UInt32(expectedCount) else {
                throw ValidationError.outOfRangeCorrection(correction.driveIndex)
            }
        }
    }
}
