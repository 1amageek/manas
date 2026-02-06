public struct MotorNervePipeline<First: MotorNerveStage, Second: MotorNerveStage>: MotorNerveStage where First.Output == Second.Input {
    public let first: First
    public let second: Second

    public init(first: First, second: Second) {
        self.first = first
        self.second = second
    }

    public func map(
        input: First.Input,
        telemetry: MotorNerveTelemetry
    ) throws -> Second.Output {
        let intermediate = try first.map(input: input, telemetry: telemetry)
        return try second.map(input: intermediate, telemetry: telemetry)
    }
}
