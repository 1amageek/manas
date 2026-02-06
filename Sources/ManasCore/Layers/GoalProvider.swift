import Foundation

/// Supplies goals to a goal-conditioned controller at each time step.
///
/// Implementations may produce constant goals (e.g., hover at altitude),
/// time-varying trajectories, or dynamically computed targets.
public protocol GoalProvider {
    /// Return the active goals for the given time.
    func currentGoals(at time: TimeInterval) -> [ControlGoal]
}
