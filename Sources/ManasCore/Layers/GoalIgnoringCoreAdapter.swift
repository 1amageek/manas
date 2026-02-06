import Foundation

/// Wraps an existing `CoreController` to satisfy `GoalConditionedCoreController`
/// by ignoring the goals and delegating to the base controller.
///
/// This adapter enables gradual migration: existing controllers work unchanged
/// in goal-conditioned pipelines until they are upgraded.
public struct GoalIgnoringCoreAdapter<Base: CoreController>: GoalConditionedCoreController {
    public var base: Base

    public init(base: Base) {
        self.base = base
    }

    public mutating func update(
        trunks: TrunkBundle,
        goals: [ControlGoal],
        time: TimeInterval
    ) throws -> [DriveIntent] {
        try base.update(trunks: trunks, time: time)
    }
}
