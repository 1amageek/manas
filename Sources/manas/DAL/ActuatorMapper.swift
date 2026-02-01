public protocol ActuatorMapper: Sendable {
    func map(
        drives: [DriveIntent],
        telemetry: DALTelemetry
    ) throws -> [ActuatorCommand]
}

