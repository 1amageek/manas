import Foundation

public struct NoActuatorLearning: ActuatorLearner {
    public init() {}

    public mutating func update(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        deltaTime: TimeInterval
    ) throws -> LearningReport? {
        _ = drives
        _ = telemetry
        _ = deltaTime
        return nil
    }
}
