import Foundation

/// A core controller that accepts goal conditioning in addition to trunk features.
///
/// This is a separate protocol from `CoreController` — existing implementations
/// are not required to adopt it. Use `GoalIgnoringCoreAdapter` to bridge a
/// `CoreController` into a `GoalConditionedCoreController`.
public protocol GoalConditionedCoreController {
    /// Produce drive intents conditioned on both trunk features and goals.
    ///
    /// - Parameters:
    ///   - trunks: Abstract feature streams from the neural hierarchy.
    ///   - goals: Active goals to condition the control policy on.
    ///   - time: Current simulation or real time.
    /// - Returns: Array of drive intents, one per drive primitive.
    mutating func update(
        trunks: TrunkBundle,
        goals: [ControlGoal],
        time: TimeInterval
    ) throws -> [DriveIntent]
}
