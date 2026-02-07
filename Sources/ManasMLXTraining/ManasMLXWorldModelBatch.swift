import MLX

/// A training batch for world model learning, containing trunk observations,
/// target drives, rewards, and episode continuation flags.
public struct ManasMLXWorldModelBatch {
    /// Input trunk features [batch, seq, inputSize].
    public let trunks: MLXArray

    /// Target drive activations from teacher [batch, seq, driveCount].
    public let targetDrives: MLXArray

    /// Dense reward signal [batch, seq, 1].
    public let rewards: MLXArray

    /// Episode continuation flags [batch, seq, 1]. 1.0 = continue, 0.0 = terminal.
    public let continues: MLXArray

    /// Descending channel values [batch, seq, K] or nil for legacy configs.
    public let descending: MLXArray?

    /// Morphology descriptor [batch, morphologyDim] or nil for legacy configs.
    public let morphology: MLXArray?

    public init(
        trunks: MLXArray,
        targetDrives: MLXArray,
        rewards: MLXArray,
        continues: MLXArray,
        descending: MLXArray? = nil,
        morphology: MLXArray? = nil
    ) {
        self.trunks = trunks
        self.targetDrives = targetDrives
        self.rewards = rewards
        self.continues = continues
        self.descending = descending
        self.morphology = morphology
    }
}
