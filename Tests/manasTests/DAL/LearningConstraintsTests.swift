import Foundation
import Testing
@testable import manas

private struct ReportingLearner: ActuatorLearner {
    var report: LearningReport?

    mutating func update(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        deltaTime: TimeInterval
    ) throws -> LearningReport? {
        _ = drives
        _ = telemetry
        _ = deltaTime
        return report
    }
}

private typealias TestDAL = DAL<IdentityActuatorMapper, ReportingLearner>

@Test func dalRejectsMissingLearningReport() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    let constraints = try LearningConstraints(
        minUpdatePeriod: 0.01,
        maxParameterDeltaNorm: 0.1,
        maxParameterDerivativeNorm: 0.1
    )
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: ReportingLearner(report: nil),
        safetyFilter: safety,
        learningConstraints: constraints
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 0.1)
    let telemetry = DALTelemetry(motors: [])

    do {
        _ = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.02)
        #expect(Bool(false))
    } catch let error as TestDAL.ValidationError {
        #expect(error == .missingLearningReport)
    }
}

@Test func dalRejectsExcessiveLearningDelta() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    let constraints = try LearningConstraints(
        minUpdatePeriod: 0.01,
        maxParameterDeltaNorm: 0.1,
        maxParameterDerivativeNorm: 10.0
    )
    let report = try LearningReport(parameterDeltaNorm: 0.2, parameterDerivativeNorm: 1.0)
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: ReportingLearner(report: report),
        safetyFilter: safety,
        learningConstraints: constraints
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 0.1)
    let telemetry = DALTelemetry(motors: [])

    do {
        _ = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.02)
        #expect(Bool(false))
    } catch let error as TestDAL.ValidationError {
        #expect(error == .parameterDeltaExceeded(0.2, 0.1))
    }
}

@Test func dalRejectsLearningUpdateTooFrequent() async throws {
    let limit = try ActuatorLimit(range: -1.0...1.0, maxRate: nil)
    let limits = ActuatorLimits(limits: [ActuatorIndex(0): limit])
    let safety = SafetyFilter(limits: limits)
    let constraints = try LearningConstraints(
        minUpdatePeriod: 0.05,
        maxParameterDeltaNorm: 1.0,
        maxParameterDerivativeNorm: 1.0
    )
    let report = try LearningReport(parameterDeltaNorm: 0.0, parameterDerivativeNorm: 0.0)
    var dal = DAL(
        mapper: IdentityActuatorMapper(),
        learner: ReportingLearner(report: report),
        safetyFilter: safety,
        learningConstraints: constraints
    )

    let drive = try DriveIntent(index: DriveIndex(0), activation: 0.1)
    let telemetry = DALTelemetry(motors: [])

    do {
        _ = try dal.update(drives: [drive], telemetry: telemetry, deltaTime: 0.01)
        #expect(Bool(false))
    } catch let error as TestDAL.ValidationError {
        #expect(error == .learningUpdateTooFrequent(0.01, 0.05))
    }
}
