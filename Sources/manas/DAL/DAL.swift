import Foundation

public struct DAL<Mapper: ActuatorMapper, Learner: ActuatorLearner>: Sendable {
    public enum ValidationError: Error, Equatable {
        case nonFiniteDelta
        case nonPositiveDelta
        case learningUpdateTooFrequent(TimeInterval, TimeInterval)
        case missingLearningReport
        case parameterDeltaExceeded(Double, Double)
        case parameterDerivativeExceeded(Double, Double)
    }

    public var mapper: Mapper
    public var learner: Learner
    public var safetyFilter: SafetyFilter
    public let learningConstraints: LearningConstraints?

    public init(
        mapper: Mapper,
        learner: Learner,
        safetyFilter: SafetyFilter,
        learningConstraints: LearningConstraints? = nil
    ) {
        self.mapper = mapper
        self.learner = learner
        self.safetyFilter = safetyFilter
        self.learningConstraints = learningConstraints
    }

    public mutating func update(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        deltaTime: TimeInterval
    ) throws -> [ActuatorCommand] {
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        if let constraints = learningConstraints {
            guard deltaTime >= constraints.minUpdatePeriod else {
                throw ValidationError.learningUpdateTooFrequent(deltaTime, constraints.minUpdatePeriod)
            }
        }

        let report = try learner.update(drives: drives, telemetry: telemetry, deltaTime: deltaTime)
        try validateLearningReport(report)
        let rawCommands = try mapper.map(drives: drives, telemetry: telemetry)
        return try safetyFilter.apply(commands: rawCommands, deltaTime: deltaTime)
    }

    private func validateLearningReport(_ report: LearningReport?) throws {
        guard let constraints = learningConstraints else {
            return
        }
        guard let report else {
            throw ValidationError.missingLearningReport
        }
        if report.parameterDeltaNorm > constraints.maxParameterDeltaNorm {
            throw ValidationError.parameterDeltaExceeded(
                report.parameterDeltaNorm,
                constraints.maxParameterDeltaNorm
            )
        }
        if report.parameterDerivativeNorm > constraints.maxParameterDerivativeNorm {
            throw ValidationError.parameterDerivativeExceeded(
                report.parameterDerivativeNorm,
                constraints.maxParameterDerivativeNorm
            )
        }
    }
}
