import Foundation

public protocol ActuatorLearner: Sendable {
    mutating func update(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        deltaTime: TimeInterval
    ) throws -> LearningReport?
}
