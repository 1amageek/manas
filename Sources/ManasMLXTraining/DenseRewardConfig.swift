/// Configuration for dense reward computation from safety and control metrics.
///
/// Reward is computed as:
/// ```
/// r_t = aliveBonus - tiltWeight * tilt² - omegaWeight * ω²
///       - effortWeight * Σ|u|² - smoothnessWeight * Σ|Δu|²
/// ```
public struct DenseRewardConfig: Sendable, Equatable, Codable {
    /// Weight for tilt angle penalty (radians²).
    public let tiltWeight: Float

    /// Weight for angular velocity penalty (rad/s²).
    public let omegaWeight: Float

    /// Weight for control effort penalty (Σ|activation|²).
    public let effortWeight: Float

    /// Weight for control smoothness penalty (Σ|Δactivation|²).
    public let smoothnessWeight: Float

    /// Constant bonus for surviving each timestep.
    public let aliveBonus: Float

    public init(
        tiltWeight: Float = 1.0,
        omegaWeight: Float = 0.1,
        effortWeight: Float = 0.01,
        smoothnessWeight: Float = 0.01,
        aliveBonus: Float = 0.1
    ) {
        self.tiltWeight = tiltWeight
        self.omegaWeight = omegaWeight
        self.effortWeight = effortWeight
        self.smoothnessWeight = smoothnessWeight
        self.aliveBonus = aliveBonus
    }
}
