import MLX
import MLXNN
import ManasMLXModels

/// World model loss computation combining drive BC, KL divergence,
/// trunk reconstruction, reward prediction, and continue prediction.
public enum WorldModelLoss {

    /// Configuration for world model loss weights.
    public struct Config: Sendable, Equatable, Codable {
        /// Weight for drive behavior cloning loss.
        public let driveWeight: Float
        /// Weight for KL divergence between posterior and prior.
        public let klWeight: Float
        /// Minimum KL in nats before gradient flows (free nats).
        public let klFreeNats: Float
        /// KL balancing ratio: higher values push prior toward posterior.
        public let klBalancing: Float
        /// Weight for trunk reconstruction loss.
        public let reconstructionWeight: Float
        /// Weight for reward prediction loss.
        public let rewardWeight: Float
        /// Weight for continue prediction loss.
        public let continueWeight: Float
        /// Number of stochastic categories for KL computation.
        public let stochasticCategories: Int
        /// Number of classes per category for KL computation.
        public let stochasticClasses: Int

        public init(
            driveWeight: Float = 1.0,
            klWeight: Float = 0.5,
            klFreeNats: Float = 1.0,
            klBalancing: Float = 0.8,
            reconstructionWeight: Float = 1.0,
            rewardWeight: Float = 1.0,
            continueWeight: Float = 1.0,
            stochasticCategories: Int = 32,
            stochasticClasses: Int = 32
        ) {
            self.driveWeight = driveWeight
            self.klWeight = klWeight
            self.klFreeNats = klFreeNats
            self.klBalancing = klBalancing
            self.reconstructionWeight = reconstructionWeight
            self.rewardWeight = rewardWeight
            self.continueWeight = continueWeight
            self.stochasticCategories = stochasticCategories
            self.stochasticClasses = stochasticClasses
        }
    }

    /// Compute the total world model loss from an RSSM forward pass.
    ///
    /// - Parameters:
    ///   - rssmOutput: Output from `ManasMLXCore.forwardRSSM()`.
    ///   - batch: Training batch with targets and rewards.
    ///   - config: Loss weight configuration.
    /// - Returns: Scalar loss value suitable for backpropagation.
    public static func compute(
        rssmOutput: RSSMOutput,
        batch: ManasMLXWorldModelBatch,
        config: Config
    ) -> MLXArray {
        let targets = ensureBatchTargets(batch.targetDrives)

        // 1. Drive BC loss
        let driveLoss = mseLoss(
            predictions: rssmOutput.drives,
            targets: targets,
            reduction: .mean
        )

        // 2. KL divergence (categorical, with free nats and balancing)
        let klLoss = categoricalKL(
            posteriorLogits: rssmOutput.posteriorLogits,
            priorLogits: rssmOutput.priorLogits,
            categories: config.stochasticCategories,
            classes: config.stochasticClasses,
            freeNats: config.klFreeNats,
            balancing: config.klBalancing
        )

        // 3. Trunk reconstruction loss
        var reconLoss = MLXArray(Float(0))
        if let trunkPred = rssmOutput.trunkPrediction {
            let trunkTargets = ensureBatchTargets(batch.trunks)
            reconLoss = mseLoss(predictions: trunkPred, targets: trunkTargets, reduction: .mean)
        }

        // 4. Reward prediction loss
        var rewardLoss = MLXArray(Float(0))
        if let rewardPred = rssmOutput.rewardPrediction {
            let rewardTargets = ensureBatchTargets(batch.rewards)
            rewardLoss = mseLoss(predictions: rewardPred, targets: rewardTargets, reduction: .mean)
        }

        // 5. Continue prediction loss (binary cross-entropy with raw logits for numerical stability)
        var continueLoss = MLXArray(Float(0))
        if let continuePred = rssmOutput.continuePrediction {
            let continueTargets = ensureBatchTargets(batch.continues)
            continueLoss = binaryCrossEntropy(
                logits: continuePred,
                targets: continueTargets,
                withLogits: true,
                reduction: .mean
            )
        }

        return config.driveWeight * driveLoss
            + config.klWeight * klLoss
            + config.reconstructionWeight * reconLoss
            + config.rewardWeight * rewardLoss
            + config.continueWeight * continueLoss
    }

    // MARK: - Categorical KL with Free Nats and Balancing

    /// Compute KL divergence between categorical posterior and prior distributions.
    ///
    /// Implements DreamerV3-style KL balancing: the prior is trained more
    /// aggressively to match the posterior than vice versa.
    /// Logits are reshaped to `[batch, seq, categories, classes]` so that softmax
    /// and KL are computed per-category, then summed across categories.
    static func categoricalKL(
        posteriorLogits: MLXArray,
        priorLogits: MLXArray,
        categories: Int,
        classes: Int,
        freeNats: Float,
        balancing: Float
    ) -> MLXArray {
        // Reshape from [batch, seq, categories*classes] to [batch, seq, categories, classes]
        let shape = posteriorLogits.shape
        let batchAndSeq = Array(shape.dropLast())
        let postReshaped = posteriorLogits.reshaped(batchAndSeq + [categories, classes])
        let priorReshaped = priorLogits.reshaped(batchAndSeq + [categories, classes])

        // Per-category softmax (over classes dimension)
        let postProbs = softmax(postReshaped, axis: -1)
        let priorProbs = softmax(priorReshaped, axis: -1)

        let eps = MLXArray(Float(1e-8))

        // KL(q || p) per category, summed over classes (axis: -1), then summed over categories (axis: -1)
        // with stopped gradient on posterior (trains prior)
        let klDyn = (stopGradient(postProbs) * (log(stopGradient(postProbs) + eps) - log(priorProbs + eps)))
            .sum(axis: -1)     // sum over classes → [batch, seq, categories]
            .sum(axis: -1)     // sum over categories → [batch, seq]
            .mean()
        let klDynClamped = maximum(klDyn, MLXArray(freeNats))

        // KL(q || p) with stopped gradient on prior (trains posterior)
        let klRep = (postProbs * (log(postProbs + eps) - log(stopGradient(priorProbs) + eps)))
            .sum(axis: -1)     // sum over classes → [batch, seq, categories]
            .sum(axis: -1)     // sum over categories → [batch, seq]
            .mean()
        let klRepClamped = maximum(klRep, MLXArray(freeNats))

        return balancing * klDynClamped + (1.0 - balancing) * klRepClamped
    }

    private static func ensureBatchTargets(_ targets: MLXArray) -> MLXArray {
        let shape = targets.shape
        guard shape.count == 2 else { return targets }
        return targets.reshaped([1, shape[0], shape[1]])
    }
}
