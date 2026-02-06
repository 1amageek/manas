import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct MLXGoalCoreController: GoalConditionedCoreController {
    public var model: ManasMLXGoalCore
    public var state: ManasMLXCoreState?
    public var activationRange: ClosedRange<Double>

    public init(
        model: ManasMLXGoalCore,
        state: ManasMLXCoreState? = nil,
        activationRange: ClosedRange<Double> = -1.0...1.0
    ) {
        self.model = model
        self.state = state
        self.activationRange = activationRange
    }

    public mutating func update(
        trunks: TrunkBundle,
        goals: [ControlGoal],
        time: TimeInterval
    ) throws -> [DriveIntent] {
        let vector = concatTrunks(trunks)
        let input = MLXArray(converting: vector.map(Double.init), [1, 1, vector.count])

        let goalVector = buildGoalVector(goals: goals)
        let goalArray = MLXArray(converting: goalVector.map(Double.init), [1, goalVector.count])

        let output = model.forward(trunks: input, goals: goalArray, state: state)
        state = output.nextState

        let lastIndex = output.drives.dim(-2) - 1
        let last = output.drives[MLXEllipsisIndex.ellipsis, lastIndex, 0...]
        eval(last)

        let values = last.asArray(Float.self)
        return values.enumerated().compactMap { index, value in
            let clamped = min(max(Double(value), activationRange.lowerBound), activationRange.upperBound)
            return try? DriveIntent(index: DriveIndex(UInt32(index)), activation: clamped)
        }
    }

    private func concatTrunks(_ bundle: TrunkBundle) -> [Float] {
        bundle.energy.map(Float.init)
        + bundle.phase.map(Float.init)
        + bundle.quality.map(Float.init)
        + bundle.spike.map(Float.init)
    }

    private func buildGoalVector(goals: [ControlGoal]) -> [Float] {
        let goalSize = model.config.goalSize
        guard let primary = goals.first else {
            return [Float](repeating: 0, count: goalSize)
        }
        let vector = primary.vector.map(Float.init)
        if vector.count >= goalSize {
            return Array(vector.prefix(goalSize))
        }
        return vector + [Float](repeating: 0, count: goalSize - vector.count)
    }
}
