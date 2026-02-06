/// The kind of goal conditioning applied to a controller.
public enum ControlGoalKind: String, Hashable, Sendable, Codable {
    /// Target state vector (e.g., desired roll, pitch, yaw, altitude).
    case targetState

    /// Trajectory as a flattened sequence of waypoints.
    case trajectory

    /// Reference signal values per drive primitive.
    case referenceSignal

    /// Learned embedding from an external encoder (e.g., text or image).
    case abstract
}
