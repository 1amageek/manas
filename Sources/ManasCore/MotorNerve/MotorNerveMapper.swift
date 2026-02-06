public protocol MotorNerveStage: Sendable {
    associatedtype Input
    associatedtype Output

    func map(
        input: Input,
        telemetry: MotorNerveTelemetry
    ) throws -> Output
}

public protocol MotorNerveMapper: MotorNerveStage where Input == [DriveIntent], Output == [ActuatorValue] {}
