import Foundation
import MLX
import MLXNN
import MLXOptimizers
import manas

public struct MLXAffineActuatorLearner: ActuatorLearner {
    public enum TelemetryFeature: CaseIterable, Sendable {
        case rpm
        case current
        case voltage
        case temperature
        case escState
    }

    public struct Configuration: Sendable, Equatable {
        public enum ValidationError: Error, Equatable {
            case emptyActuatorIndices
            case duplicateActuatorIndex(ActuatorIndex)
            case invalidRange(String)
            case nonPositiveDeltaMax
            case negativeLearningRate
            case missingTelemetryRange(TelemetryFeature)
        }

        public let actuatorIndices: [ActuatorIndex]
        public let telemetryFeatures: [TelemetryFeature]
        public let driveRange: ClosedRange<Double>
        public let telemetryRanges: [TelemetryFeature: ClosedRange<Double>]
        public let deltaMax: Double
        public let learningRate: Double
        public let targetTelemetry: TelemetryFeature?
        public let enabled: Bool

        public init(
            actuatorIndices: [ActuatorIndex],
            telemetryFeatures: [TelemetryFeature],
            driveRange: ClosedRange<Double>,
            telemetryRanges: [TelemetryFeature: ClosedRange<Double>],
            deltaMax: Double,
            learningRate: Double,
            targetTelemetry: TelemetryFeature?,
            enabled: Bool
        ) throws {
            guard !actuatorIndices.isEmpty else { throw ValidationError.emptyActuatorIndices }
            let indexSet = Set(actuatorIndices)
            if indexSet.count != actuatorIndices.count {
                if let duplicate = actuatorIndices.first(where: { index in
                    actuatorIndices.filter { $0 == index }.count > 1
                }) {
                    throw ValidationError.duplicateActuatorIndex(duplicate)
                }
            }
            try Configuration.validateRange(driveRange, field: "driveRange")
            for feature in telemetryFeatures {
                guard let range = telemetryRanges[feature] else {
                    throw ValidationError.missingTelemetryRange(feature)
                }
                try Configuration.validateRange(range, field: "telemetryRanges.\(feature)")
            }
            if let targetTelemetry {
                guard let range = telemetryRanges[targetTelemetry] else {
                    throw ValidationError.missingTelemetryRange(targetTelemetry)
                }
                try Configuration.validateRange(range, field: "telemetryRanges.\(targetTelemetry)")
            }
            guard deltaMax.isFinite, deltaMax > 0 else {
                throw ValidationError.nonPositiveDeltaMax
            }
            guard learningRate.isFinite, learningRate >= 0 else {
                throw ValidationError.negativeLearningRate
            }

            self.actuatorIndices = actuatorIndices
            self.telemetryFeatures = telemetryFeatures
            self.driveRange = driveRange
            self.telemetryRanges = telemetryRanges
            self.deltaMax = deltaMax
            self.learningRate = learningRate
            self.targetTelemetry = targetTelemetry
            self.enabled = enabled
        }

        public init(
            actuatorCount: Int,
            telemetryFeatures: [TelemetryFeature],
            driveRange: ClosedRange<Double>,
            telemetryRanges: [TelemetryFeature: ClosedRange<Double>],
            deltaMax: Double,
            learningRate: Double,
            targetTelemetry: TelemetryFeature?,
            enabled: Bool
        ) throws {
            let indices = (0..<max(actuatorCount, 0)).map { ActuatorIndex(UInt32($0)) }
            try self.init(
                actuatorIndices: indices,
                telemetryFeatures: telemetryFeatures,
                driveRange: driveRange,
                telemetryRanges: telemetryRanges,
                deltaMax: deltaMax,
                learningRate: learningRate,
                targetTelemetry: targetTelemetry,
                enabled: enabled
            )
        }

        private static func validateRange(_ range: ClosedRange<Double>, field: String) throws {
            guard range.lowerBound.isFinite, range.upperBound.isFinite else {
                throw ValidationError.invalidRange(field)
            }
            guard range.upperBound > range.lowerBound else {
                throw ValidationError.invalidRange(field)
            }
        }
    }

    public enum ValidationError: Error, Equatable {
        case missingBaseCommand(ActuatorIndex)
    }

    public var configuration: Configuration
    private var model: AffineDeltaModel
    private var optimizer: SGD

    public init(configuration: Configuration) {
        self.configuration = configuration
        let featureCount = 1 + configuration.telemetryFeatures.count
        self.model = AffineDeltaModel(featureCount: featureCount)
        self.optimizer = SGD(learningRate: Float(configuration.learningRate))
        eval(model)
    }

    public mutating func infer(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        baseCommands: [ActuatorCommand]
    ) throws -> [ActuatorCommandDelta] {
        let baseIndexSet = Set(baseCommands.map(\.index))
        for index in configuration.actuatorIndices where !baseIndexSet.contains(index) {
            throw ValidationError.missingBaseCommand(index)
        }

        let featureArray = buildFeatureArray(drives: drives, telemetry: telemetry)
        let rawOutputs = model(featureArray)
        let bounded = tanh(rawOutputs) * MLXArray(Float(configuration.deltaMax))
        eval(bounded)
        let rawValues = bounded.asArray(Float.self)

        return try zip(configuration.actuatorIndices, rawValues).map { index, raw in
            return try ActuatorCommandDelta(index: index, value: Double(raw))
        }
    }

    public mutating func update(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        deltaTime: TimeInterval
    ) throws -> LearningReport? {
        guard configuration.enabled else {
            return try LearningReport(parameterDeltaNorm: 0, parameterDerivativeNorm: 0)
        }
        guard let targetTelemetry = configuration.targetTelemetry else {
            return try LearningReport(parameterDeltaNorm: 0, parameterDerivativeNorm: 0)
        }

        let featureArray = buildFeatureArray(drives: drives, telemetry: telemetry)
        let targets = buildTargetDeltas(
            drives: drives,
            telemetry: telemetry,
            targetFeature: targetTelemetry
        )

        guard let targets else {
            return try LearningReport(parameterDeltaNorm: 0, parameterDerivativeNorm: 0)
        }

        let targetArray = MLXArray(targets, [configuration.actuatorIndices.count, 1])
        let deltaMax = MLXArray(Float(configuration.deltaMax))

        let previousWeights = model.linear.weight.asArray(Float.self)
        let previousBias = model.linear.bias?.asArray(Float.self) ?? []

        func loss(model: AffineDeltaModel, x: MLXArray, y: MLXArray) -> MLXArray {
            let prediction = tanh(model(x)) * deltaMax
            let error = prediction - y
            return mean(square(error))
        }

        let lossAndGrad = valueAndGrad(model: model, loss)
        let (_, grads) = lossAndGrad(model, featureArray, targetArray)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)

        let currentWeights = model.linear.weight.asArray(Float.self)
        let currentBias = model.linear.bias?.asArray(Float.self) ?? []
        let deltaNorm = parameterDeltaNorm(
            previousWeights: previousWeights,
            previousBias: previousBias,
            currentWeights: currentWeights,
            currentBias: currentBias
        )
        let derivativeNorm = deltaNorm / deltaTime

        return try LearningReport(
            parameterDeltaNorm: deltaNorm,
            parameterDerivativeNorm: derivativeNorm
        )
    }

    private func buildFeatureArray(
        drives: [DriveIntent],
        telemetry: DALTelemetry
    ) -> MLXArray {
        let driveByIndex = Dictionary(
            uniqueKeysWithValues: drives.map { (ActuatorIndex($0.index.rawValue), $0.activation) }
        )
        let telemetryByIndex = Dictionary(
            uniqueKeysWithValues: telemetry.motors.map { ($0.index, $0) }
        )

        let featureCount = 1 + configuration.telemetryFeatures.count
        var features: [Float] = []
        features.reserveCapacity(configuration.actuatorIndices.count * featureCount)

        for index in configuration.actuatorIndices {
            let driveValue = driveByIndex[index] ?? 0.0
            let driveNorm = normalize(value: driveValue, range: configuration.driveRange)
            features.append(Float(driveNorm))

            let motorTelemetry = telemetryByIndex[index]
            for feature in configuration.telemetryFeatures {
                let value = telemetryValue(feature, from: motorTelemetry)
                let range = configuration.telemetryRanges[feature] ?? 0.0...1.0
                let normalized = value.map { normalize(value: $0, range: range) } ?? 0.0
                features.append(Float(normalized))
            }
        }

        return MLXArray(features, [configuration.actuatorIndices.count, featureCount])
    }

    private func buildTargetDeltas(
        drives: [DriveIntent],
        telemetry: DALTelemetry,
        targetFeature: TelemetryFeature
    ) -> [Float]? {
        let driveByIndex = Dictionary(
            uniqueKeysWithValues: drives.map { (ActuatorIndex($0.index.rawValue), $0.activation) }
        )
        let telemetryByIndex = Dictionary(
            uniqueKeysWithValues: telemetry.motors.map { ($0.index, $0) }
        )

        guard let range = configuration.telemetryRanges[targetFeature] else {
            return nil
        }

        var targets: [Float] = []
        targets.reserveCapacity(configuration.actuatorIndices.count)

        for index in configuration.actuatorIndices {
            guard let driveValue = driveByIndex[index] else {
                return nil
            }
            guard let motorTelemetry = telemetryByIndex[index] else {
                return nil
            }
            guard let measured = telemetryValue(targetFeature, from: motorTelemetry) else {
                return nil
            }

            let desired = normalize(value: driveValue, range: configuration.driveRange)
            let observed = normalize(value: measured, range: range)
            let error = max(-1.0, min(1.0, desired - observed))
        let target = error * configuration.deltaMax
        targets.append(Float(target))
    }

    return targets
}

    private func telemetryValue(_ feature: TelemetryFeature, from telemetry: MotorTelemetry?) -> Double? {
        guard let telemetry else { return nil }
        switch feature {
        case .rpm:
            return telemetry.rpm
        case .current:
            return telemetry.current
        case .voltage:
            return telemetry.voltage
        case .temperature:
            return telemetry.temperature
        case .escState:
            return telemetry.escState
        }
    }

    private func normalize(value: Double, range: ClosedRange<Double>) -> Double {
        let clamped = min(max(value, range.lowerBound), range.upperBound)
        let span = range.upperBound - range.lowerBound
        let scaled = (clamped - range.lowerBound) / span
        return (scaled * 2.0) - 1.0
    }

    private func parameterDeltaNorm(
        previousWeights: [Float],
        previousBias: [Float],
        currentWeights: [Float],
        currentBias: [Float]
    ) -> Double {
        var sumSquares = 0.0
        for (current, previous) in zip(currentWeights, previousWeights) {
            let delta = Double(current - previous)
            sumSquares += delta * delta
        }
        for (current, previous) in zip(currentBias, previousBias) {
            let delta = Double(current - previous)
            sumSquares += delta * delta
        }
        return sqrt(sumSquares)
    }
}
