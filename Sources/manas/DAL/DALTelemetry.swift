public struct DALTelemetry: Sendable, Codable, Equatable {
    public let motors: [MotorTelemetry]

    public init(motors: [MotorTelemetry]) {
        self.motors = motors
    }
}

