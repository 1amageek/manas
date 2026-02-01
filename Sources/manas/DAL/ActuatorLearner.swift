import Foundation

public protocol ActuatorLearner {
    mutating func infer(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        baseCommands: [ActuatorCommand]
    ) throws -> [ActuatorCommandDelta]

    mutating func update(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        deltaTime: TimeInterval
    ) throws -> LearningReport?
}

public extension ActuatorLearner {
    mutating func infer(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        baseCommands: [ActuatorCommand]
    ) throws -> [ActuatorCommandDelta] {
        _ = drives
        _ = telemetry
        _ = baseCommands
        return []
    }
}
