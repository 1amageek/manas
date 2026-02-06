public struct IdentityMotorNerveMapper: MotorNerveMapper {
    public init() {}

    public func map(
        input drives: [DriveIntent],
        telemetry: MotorNerveTelemetry
    ) throws -> [ActuatorValue] {
        _ = telemetry
        return try drives.map { drive in
            let index = ActuatorIndex(drive.index.rawValue)
            return try ActuatorValue(index: index, value: drive.activation)
        }
    }
}
