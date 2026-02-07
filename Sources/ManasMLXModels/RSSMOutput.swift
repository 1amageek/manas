import MLX

/// Output from the RSSM forward pass, containing both policy outputs
/// and world model predictions for training.
public struct RSSMOutput {
    /// Policy-derived drive signals [batch, seq, driveCount].
    public let drives: MLXArray

    /// Posterior logits q(z_t | h_t, x_t) [batch, seq, categories * classes].
    public let posteriorLogits: MLXArray

    /// Prior logits p(z_t | h_t) [batch, seq, categories * classes].
    public let priorLogits: MLXArray

    /// Sampled posterior latent [batch, seq, categories * classes].
    public let posteriorZ: MLXArray

    /// Reconstructed current trunk features [batch, seq, inputSize].
    public let trunkPrediction: MLXArray?

    /// Predicted reward [batch, seq, 1].
    public let rewardPrediction: MLXArray?

    /// Predicted episode continuation probability [batch, seq, 1].
    public let continuePrediction: MLXArray?

    /// Updated recurrent state for the next timestep.
    public let nextState: ManasMLXCoreState

    public init(
        drives: MLXArray,
        posteriorLogits: MLXArray,
        priorLogits: MLXArray,
        posteriorZ: MLXArray,
        trunkPrediction: MLXArray?,
        rewardPrediction: MLXArray?,
        continuePrediction: MLXArray?,
        nextState: ManasMLXCoreState
    ) {
        self.drives = drives
        self.posteriorLogits = posteriorLogits
        self.priorLogits = priorLogits
        self.posteriorZ = posteriorZ
        self.trunkPrediction = trunkPrediction
        self.rewardPrediction = rewardPrediction
        self.continuePrediction = continuePrediction
        self.nextState = nextState
    }
}
