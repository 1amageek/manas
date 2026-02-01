public struct MotorTelemetry: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
    }

    public let index: ActuatorIndex
    public let rpm: Double?
    public let current: Double?
    public let voltage: Double?
    public let temperature: Double?
    public let escState: Double?

    public init(
        index: ActuatorIndex,
        rpm: Double?,
        current: Double?,
        voltage: Double?,
        temperature: Double?,
        escState: Double?
    ) throws {
        try MotorTelemetry.validateOptionalNonNegativeFinite(rpm, field: "rpm")
        try MotorTelemetry.validateOptionalNonNegativeFinite(current, field: "current")
        try MotorTelemetry.validateOptionalNonNegativeFinite(voltage, field: "voltage")
        try MotorTelemetry.validateOptionalNonNegativeFinite(temperature, field: "temperature")
        try MotorTelemetry.validateOptionalFinite(escState, field: "escState")

        self.index = index
        self.rpm = rpm
        self.current = current
        self.voltage = voltage
        self.temperature = temperature
        self.escState = escState
    }

    private static func validateOptionalNonNegativeFinite(_ value: Double?, field: String) throws {
        guard let value else { return }
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }

    private static func validateOptionalFinite(_ value: Double?, field: String) throws {
        guard let value else { return }
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
    }
}

