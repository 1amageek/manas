import Foundation
import MLX
import ManasCore
import ManasMLXModels

public struct MLXDescendingCoreController: GoalConditionedCoreController {
    public enum ControllerError: Error, Equatable {
        case nonFiniteOutput(index: Int)
    }

    public var model: ManasMLXCore
    public var state: ManasMLXCoreState?
    public var activationRange: ClosedRange<Double>

    public init(
        model: ManasMLXCore,
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
        let input = ManasMLXRuntimeTensorInput.trunkInput(from: trunks)

        let descending: MLXArray?
        if model.config.descendingEnabled {
            descending = ManasMLXRuntimeTensorInput.descendingInput(
                goals: goals,
                size: model.config.descendingSize
            )
        } else {
            descending = nil
        }

        let output = model.forward(trunks: input, descending: descending, state: state)
        state = output.nextState

        let lastIndex = output.drives.dim(-2) - 1
        let last = output.drives[MLXEllipsisIndex.ellipsis, lastIndex, 0...]
        eval(last)

        let values = last.asArray(Float.self)
        return try values.enumerated().map { index, value in
            let clamped = min(max(Double(value), activationRange.lowerBound), activationRange.upperBound)
            guard clamped.isFinite else {
                throw ControllerError.nonFiniteOutput(index: index)
            }
            return try DriveIntent(index: DriveIndex(UInt32(index)), activation: clamped)
        }
    }
}
