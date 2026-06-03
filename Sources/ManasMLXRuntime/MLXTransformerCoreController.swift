import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct MLXTransformerCoreController: CoreController, GoalConditionedCoreController {
    public enum ControllerError: Error, Equatable {
        case nonFiniteOutput(index: Int)
    }

    public var model: ManasMLXTransformerCore
    public var activationRange: ClosedRange<Double>

    private var trunkHistory: [[Float]]
    private let maxSequenceLength: Int

    public init(
        model: ManasMLXTransformerCore,
        activationRange: ClosedRange<Double> = -1.0...1.0
    ) {
        self.model = model
        self.activationRange = activationRange
        self.maxSequenceLength = model.config.maxSequenceLength
        self.trunkHistory = []
    }

    public mutating func update(trunks: TrunkBundle, time: TimeInterval) throws -> [DriveIntent] {
        let vector = ManasMLXRuntimeTensorInput.trunkVector(from: trunks)
        trunkHistory.append(vector)
        if trunkHistory.count > maxSequenceLength {
            trunkHistory.removeFirst(trunkHistory.count - maxSequenceLength)
        }

        let seqLen = trunkHistory.count
        let inputSize = vector.count
        let input = ManasMLXRuntimeTensorInput.trunkHistoryInput(history: trunkHistory, inputSize: inputSize)

        let output = model.forward(trunks: input)
        let lastIndex = output.drives.dim(-2) - 1
        let last = output.drives[MLXEllipsisIndex.ellipsis, lastIndex, 0...]
        eval(last)

        let values = last.asArray(Float.self)
        return try buildDriveIntents(values)
    }

    public mutating func update(
        trunks: TrunkBundle,
        goals: [ControlGoal],
        time: TimeInterval
    ) throws -> [DriveIntent] {
        guard let goalSize = model.config.goalSize, !goals.isEmpty else {
            return try update(trunks: trunks, time: time)
        }

        let vector = ManasMLXRuntimeTensorInput.trunkVector(from: trunks)
        trunkHistory.append(vector)
        if trunkHistory.count > maxSequenceLength {
            trunkHistory.removeFirst(trunkHistory.count - maxSequenceLength)
        }

        let seqLen = trunkHistory.count
        let inputSize = vector.count
        let input = ManasMLXRuntimeTensorInput.trunkHistoryInput(history: trunkHistory, inputSize: inputSize)
        let goalArray = ManasMLXRuntimeTensorInput.goalInput(goals: goals, size: goalSize)

        let output = model.forward(trunks: input, goals: goalArray)
        let lastIndex = output.drives.dim(-2) - 1
        let last = output.drives[MLXEllipsisIndex.ellipsis, lastIndex, 0...]
        eval(last)

        let values = last.asArray(Float.self)
        return try buildDriveIntents(values)
    }

    public mutating func reset() {
        trunkHistory.removeAll()
    }

    private func buildDriveIntents(_ values: [Float]) throws -> [DriveIntent] {
        try values.enumerated().map { index, value in
            let clamped = min(max(Double(value), activationRange.lowerBound), activationRange.upperBound)
            guard clamped.isFinite else {
                throw ControllerError.nonFiniteOutput(index: index)
            }
            return try DriveIntent(index: DriveIndex(UInt32(index)), activation: clamped)
        }
    }
}
