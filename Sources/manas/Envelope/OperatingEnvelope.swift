public struct OperatingEnvelope: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case emptyIdentifier
        case emptyVersion
    }

    public let id: String
    public let version: String
    public let physical: PhysicalBounds
    public let sensor: SensorBounds
    public let disturbance: DisturbanceBounds
    public let actuator: ActuatorBounds
    public let updateRates: UpdateRates

    public init(
        id: String,
        version: String,
        physical: PhysicalBounds,
        sensor: SensorBounds,
        disturbance: DisturbanceBounds,
        actuator: ActuatorBounds,
        updateRates: UpdateRates
    ) throws {
        guard !id.isEmpty else {
            throw ValidationError.emptyIdentifier
        }
        guard !version.isEmpty else {
            throw ValidationError.emptyVersion
        }

        self.id = id
        self.version = version
        self.physical = physical
        self.sensor = sensor
        self.disturbance = disturbance
        self.actuator = actuator
        self.updateRates = updateRates
    }
}
