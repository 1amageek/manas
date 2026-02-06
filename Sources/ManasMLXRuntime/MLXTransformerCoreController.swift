import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct MLXTransformerCoreController: CoreController, GoalConditionedCoreController {
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
        let vector = concatTrunks(trunks)
        trunkHistory.append(vector)
        if trunkHistory.count > maxSequenceLength {
            trunkHistory.removeFirst(trunkHistory.count - maxSequenceLength)
        }

        let seqLen = trunkHistory.count
        let inputSize = vector.count
        let flat = trunkHistory.flatMap { $0 }
        let input = MLXArray(converting: flat.map(Double.init), [1, seqLen, inputSize])

        let output = model.forward(trunks: input)
        let lastIndex = output.drives.dim(-2) - 1
        let last = output.drives[MLXEllipsisIndex.ellipsis, lastIndex, 0...]
        eval(last)

        let values = last.asArray(Float.self)
        return values.enumerated().compactMap { index, value in
            let clamped = min(max(Double(value), activationRange.lowerBound), activationRange.upperBound)
            return try? DriveIntent(index: DriveIndex(UInt32(index)), activation: clamped)
        }
    }

    public mutating func update(
        trunks: TrunkBundle,
        goals: [ControlGoal],
        time: TimeInterval
    ) throws -> [DriveIntent] {
        guard let goalSize = model.config.goalSize, !goals.isEmpty else {
            return try update(trunks: trunks, time: time)
        }

        let vector = concatTrunks(trunks)
        trunkHistory.append(vector)
        if trunkHistory.count > maxSequenceLength {
            trunkHistory.removeFirst(trunkHistory.count - maxSequenceLength)
        }

        let seqLen = trunkHistory.count
        let inputSize = vector.count
        let flat = trunkHistory.flatMap { $0 }
        let input = MLXArray(converting: flat.map(Double.init), [1, seqLen, inputSize])

        let goalVector = goals.first.map { $0.vector.map(Float.init) }
            ?? [Float](repeating: 0, count: goalSize)
        let paddedGoal = Array(goalVector.prefix(goalSize))
            + [Float](repeating: 0, count: max(0, goalSize - goalVector.count))
        let goalArray = MLXArray(converting: paddedGoal.map(Double.init), [1, 1, goalSize])

        let output = model.forward(trunks: input, goals: goalArray)
        let lastIndex = output.drives.dim(-2) - 1
        let last = output.drives[MLXEllipsisIndex.ellipsis, lastIndex, 0...]
        eval(last)

        let values = last.asArray(Float.self)
        return values.enumerated().compactMap { index, value in
            let clamped = min(max(Double(value), activationRange.lowerBound), activationRange.upperBound)
            return try? DriveIntent(index: DriveIndex(UInt32(index)), activation: clamped)
        }
    }

    public mutating func reset() {
        trunkHistory.removeAll()
    }

    private func concatTrunks(_ bundle: TrunkBundle) -> [Float] {
        bundle.energy.map(Float.init)
        + bundle.phase.map(Float.init)
        + bundle.quality.map(Float.init)
        + bundle.spike.map(Float.init)
    }
}
