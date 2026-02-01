public struct IdentityActuatorMapper: ActuatorMapper {
    public init() {}

    public func map(
        drives: [DriveIntent],
        telemetry: DALTelemetry
    ) throws -> [ActuatorCommand] {
        try drives.map { drive in
            let index = ActuatorIndex(drive.index.rawValue)
            return try ActuatorCommand(index: index, value: drive.activation)
        }
    }
}

