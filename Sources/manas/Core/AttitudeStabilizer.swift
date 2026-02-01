public struct AttitudeStabilizer: DriveSynthesizer {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
    }

    public struct Gains: Sendable, Codable, Equatable {
        public enum ValidationError: Error, Equatable {
            case nonFinite(String)
            case negative(String)
        }

        public let kp: Double
        public let kd: Double

        public init(kp: Double, kd: Double) throws {
            guard kp.isFinite else { throw ValidationError.nonFinite("kp") }
            guard kd.isFinite else { throw ValidationError.nonFinite("kd") }
            guard kp >= 0 else { throw ValidationError.negative("kp") }
            guard kd >= 0 else { throw ValidationError.negative("kd") }

            self.kp = kp
            self.kd = kd
        }
    }

    public let driveRoll: DriveIndex
    public let drivePitch: DriveIndex
    public let driveYaw: DriveIndex
    public let maxTorque: Double
    public let maxAngularRate: Double
    public let mapping: IMU6Mapping
    public let gains: Gains

    public init(
        driveRoll: DriveIndex,
        drivePitch: DriveIndex,
        driveYaw: DriveIndex,
        maxTorque: Double,
        maxAngularRate: Double,
        mapping: IMU6Mapping,
        gains: Gains
    ) throws {
        guard maxTorque.isFinite else { throw ValidationError.nonFinite("maxTorque") }
        guard maxAngularRate.isFinite else { throw ValidationError.nonFinite("maxAngularRate") }
        guard maxTorque >= 0 else { throw ValidationError.negative("maxTorque") }
        guard maxAngularRate >= 0 else { throw ValidationError.negative("maxAngularRate") }

        self.driveRoll = driveRoll
        self.drivePitch = drivePitch
        self.driveYaw = driveYaw
        self.maxTorque = maxTorque
        self.maxAngularRate = maxAngularRate
        self.mapping = mapping
        self.gains = gains
    }

    public func synthesize(
        energies: [EnergyState],
        phases: [PhaseState],
        regime: Regime
    ) throws -> [DriveIntent] {
        _ = energies
        _ = regime

        let vectors = try PhaseAccess.vector(phases: phases, mapping: mapping)
        let gyro = vectors.gyro.clamped(maxMagnitude: maxAngularRate)
        let accel = vectors.accel.clamped(maxMagnitude: maxAngularRate)

        let rollTorque = -gains.kp * accel.y - gains.kd * gyro.x
        let pitchTorque = gains.kp * accel.x - gains.kd * gyro.y
        let yawTorque = -gains.kd * gyro.z

        let roll = clamp(rollTorque, limit: maxTorque)
        let pitch = clamp(pitchTorque, limit: maxTorque)
        let yaw = clamp(yawTorque, limit: maxTorque)

        return [
            try DriveIntent(index: driveRoll, activation: roll),
            try DriveIntent(index: drivePitch, activation: pitch),
            try DriveIntent(index: driveYaw, activation: yaw),
        ]
    }

    private func clamp(_ value: Double, limit: Double) -> Double {
        let clamped = min(max(value, -limit), limit)
        return clamped.isFinite ? clamped : 0.0
    }
}
