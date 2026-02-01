public struct PhaseAccess {
    public enum ValidationError: Error, Equatable {
        case missing(PhaseIndex)
    }

    public static func vector(
        phases: [PhaseState],
        mapping: IMU6Mapping
    ) throws -> (gyro: SIMD3<Double>, accel: SIMD3<Double>) {
        var lookup: [PhaseIndex: Double] = [:]
        for phase in phases {
            lookup[phase.index] = phase.value
        }

        guard let gx = lookup[mapping.gyroX] else { throw ValidationError.missing(mapping.gyroX) }
        guard let gy = lookup[mapping.gyroY] else { throw ValidationError.missing(mapping.gyroY) }
        guard let gz = lookup[mapping.gyroZ] else { throw ValidationError.missing(mapping.gyroZ) }
        guard let ax = lookup[mapping.accelX] else { throw ValidationError.missing(mapping.accelX) }
        guard let ay = lookup[mapping.accelY] else { throw ValidationError.missing(mapping.accelY) }
        guard let az = lookup[mapping.accelZ] else { throw ValidationError.missing(mapping.accelZ) }

        return (
            gyro: SIMD3<Double>(gx, gy, gz),
            accel: SIMD3<Double>(ax, ay, az)
        )
    }
}
