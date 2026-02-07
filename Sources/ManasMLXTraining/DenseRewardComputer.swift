/// Computes dense reward from safety and control metrics.
///
/// The reward function penalizes unsafe states and control inefficiency:
/// ```
/// r_t = aliveBonus - tiltWeight * tilt² - omegaWeight * ω²
///       - effortWeight * Σ|u|² - smoothnessWeight * Σ|Δu|²
/// ```
public enum DenseRewardComputer {

    /// Compute a scalar reward from safety metrics and drive outputs.
    ///
    /// - Parameters:
    ///   - tiltRadians: Body tilt from vertical (radians).
    ///   - omegaMagnitude: Angular velocity magnitude (rad/s).
    ///   - driveActivations: Current drive activation values.
    ///   - previousDriveActivations: Previous step's drive activations (for smoothness).
    ///   - config: Reward weight configuration.
    /// - Returns: Scalar reward value.
    public static func computeReward(
        tiltRadians: Float,
        omegaMagnitude: Float,
        driveActivations: [Float],
        previousDriveActivations: [Float]?,
        config: DenseRewardConfig
    ) -> Float {
        let tiltPenalty = config.tiltWeight * tiltRadians * tiltRadians
        let omegaPenalty = config.omegaWeight * omegaMagnitude * omegaMagnitude

        var effortSum: Float = 0
        for activation in driveActivations {
            effortSum += activation * activation
        }
        let effortPenalty = config.effortWeight * effortSum

        var smoothnessPenalty: Float = 0
        if let previous = previousDriveActivations {
            for (current, prev) in zip(driveActivations, previous) {
                let delta = current - prev
                smoothnessPenalty += delta * delta
            }
            smoothnessPenalty *= config.smoothnessWeight
        }

        return config.aliveBonus - tiltPenalty - omegaPenalty - effortPenalty - smoothnessPenalty
    }
}
