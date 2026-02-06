/// A goal condition that specifies what the controller should achieve.
///
/// Inspired by Micro-World's text/image conditioning, this enables
/// controllers to receive target states, trajectories, or abstract
/// embeddings as conditioning inputs.
public struct ControlGoal: Sendable, Equatable {
    public enum ValidationError: Error, Equatable {
        case emptyVector
        case nonFiniteValue
    }

    /// The kind of goal (target state, trajectory, reference signal, or abstract).
    public let kind: ControlGoalKind

    /// The goal vector. Interpretation depends on `kind`.
    public let vector: [Double]

    /// Priority of this goal relative to others. Clamped to [0, 1].
    public let priority: Double

    public init(kind: ControlGoalKind, vector: [Double], priority: Double = 1.0) throws {
        guard !vector.isEmpty else {
            throw ValidationError.emptyVector
        }
        guard vector.allSatisfy({ $0.isFinite }), priority.isFinite else {
            throw ValidationError.nonFiniteValue
        }
        self.kind = kind
        self.vector = vector
        self.priority = min(max(priority, 0.0), 1.0)
    }
}
