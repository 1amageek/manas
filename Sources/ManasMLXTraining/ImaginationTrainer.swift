import MLX
import MLXNN
import MLXOptimizers
import ManasMLXModels

/// Trains the actor (driveHead) and critic (valueHead) using imagination
/// rollouts within the learned world model. The world model parameters
/// (encoder, GRUs, posterior, prior, prediction heads) are frozen during
/// imagination training.
public enum ImaginationTrainer {

    /// Configuration for imagination-based actor-critic training.
    public struct Config: Sendable, Equatable, Codable {
        /// Number of imagination rollout steps.
        public let horizon: Int
        /// Discount factor for returns.
        public let gamma: Float
        /// Lambda for generalized advantage estimation.
        public let lambda: Float
        /// Learning rate for actor (driveHead) optimizer.
        public let actorLearningRate: Float
        /// Learning rate for critic (valueHead) optimizer.
        public let criticLearningRate: Float
        /// Entropy bonus coefficient for exploration.
        public let entropyBonus: Float
        /// Maximum gradient norm for clipping.
        public let maxGradNorm: Float?

        public init(
            horizon: Int = 15,
            gamma: Float = 0.997,
            lambda: Float = 0.95,
            actorLearningRate: Float = 3e-5,
            criticLearningRate: Float = 3e-5,
            entropyBonus: Float = 3e-4,
            maxGradNorm: Float? = 100.0
        ) {
            self.horizon = horizon
            self.gamma = gamma
            self.lambda = lambda
            self.actorLearningRate = actorLearningRate
            self.criticLearningRate = criticLearningRate
            self.entropyBonus = entropyBonus
            self.maxGradNorm = maxGradNorm
        }
    }

    /// Train actor and critic using imagination rollouts from collected start states.
    ///
    /// - Parameters:
    ///   - model: The ManasMLXCore with RSSM enabled.
    ///   - startStates: Initial states collected from real environment interaction.
    ///   - config: Imagination training configuration.
    ///   - epochs: Number of training epochs over the start states.
    /// - Returns: Training result with loss histories and reward statistics.
    public static func train(
        model: ManasMLXCore,
        startStates: [ManasMLXCoreState],
        config: Config,
        epochs: Int
    ) -> ImaginationTrainingResult {
        guard !startStates.isEmpty, model.config.rssmEnabled, model.config.valueHeadHiddenSize > 0 else {
            return ImaginationTrainingResult(
                actorLosses: [], criticLosses: [],
                meanReward: 0, meanHorizonLength: 0
            )
        }

        let actorOptimizer = Adam(learningRate: config.actorLearningRate)
        let criticOptimizer = Adam(learningRate: config.criticLearningRate)

        var actorLosses: [Float] = []
        var criticLosses: [Float] = []
        var totalReward: Float = 0
        var totalHorizon: Float = 0
        var rolloutCount: Float = 0

        for _ in 0..<epochs {
            var epochActorLoss: Float = 0
            var epochCriticLoss: Float = 0

            for startState in startStates {
                // Freeze world model, keep actor/critic trainable
                freezeWorldModel(model)

                // Imagination rollout
                let rollout = rolloutImagination(
                    model: model, startState: startState, config: config
                )

                // Compute lambda-returns (used for statistics only)
                _ = computeLambdaReturns(
                    rewards: rollout.rewards,
                    values: rollout.values,
                    continues: rollout.continues,
                    gamma: config.gamma,
                    lambda: config.lambda
                )

                // Actor loss: maximize returns (minimize negative returns) + entropy regularization
                let actorLg = valueAndGrad(model: model) { model, _, _ in
                    let innerRollout = rolloutImagination(
                        model: model, startState: startState, config: config
                    )
                    let innerReturns = computeLambdaReturns(
                        rewards: innerRollout.rewards,
                        values: innerRollout.values.map { stopGradient($0) },
                        continues: innerRollout.continues,
                        gamma: config.gamma,
                        lambda: config.lambda
                    )
                    let meanReturn = stackedMean(innerReturns)
                    // Entropy proxy for deterministic policy: penalize tanh saturation
                    // log(1 - tanh²(x)) is maximized when x is near 0 (unsaturated)
                    var entropyProxy = MLXArray(Float(0))
                    if config.entropyBonus > 0 {
                        let logUnsaturated = innerRollout.rawActions.map { raw in
                            log(1.0 - tanh(raw).square() + 1e-6).mean()
                        }
                        entropyProxy = stackedMean(logUnsaturated)
                    }
                    return -meanReturn - config.entropyBonus * entropyProxy
                }

                let dummyInput = MLXArray.zeros([1, 1])
                let (actorLoss, actorGrads) = actorLg(model, dummyInput, dummyInput)
                let clippedActorGrads = clipIfNeeded(actorGrads, maxNorm: config.maxGradNorm)
                actorOptimizer.update(model: model, gradients: clippedActorGrads)
                eval(model, actorOptimizer)

                // Critic loss: MSE(value, target_return)
                // Freeze world model AND driveHead so only valueHead is updated
                unfreezeAll(model)
                freezeWorldModelAndActor(model)

                let criticLg = valueAndGrad(model: model) { model, _, _ in
                    let innerRollout = rolloutImagination(
                        model: model, startState: startState, config: config
                    )
                    let targetReturns = computeLambdaReturns(
                        rewards: innerRollout.rewards.map { stopGradient($0) },
                        values: innerRollout.values,
                        continues: innerRollout.continues.map { stopGradient($0) },
                        gamma: config.gamma,
                        lambda: config.lambda
                    )
                    var criticLoss = MLXArray(Float(0))
                    for (i, value) in innerRollout.values.dropLast().enumerated() {
                        let target = stopGradient(targetReturns[i])
                        criticLoss = criticLoss + mseLoss(predictions: value, targets: target, reduction: .mean)
                    }
                    return criticLoss / Float(max(innerRollout.values.count - 1, 1))
                }

                let (criticLoss, criticGrads) = criticLg(model, dummyInput, dummyInput)
                let clippedCriticGrads = clipIfNeeded(criticGrads, maxNorm: config.maxGradNorm)
                criticOptimizer.update(model: model, gradients: clippedCriticGrads)
                eval(model, criticOptimizer)

                unfreezeAll(model)

                epochActorLoss += actorLoss.item(Float.self)
                epochCriticLoss += criticLoss.item(Float.self)

                // Collect statistics
                for reward in rollout.rewards {
                    totalReward += reward.mean().item(Float.self)
                }
                totalHorizon += Float(rollout.rewards.count)
                rolloutCount += 1
            }

            actorLosses.append(epochActorLoss / Float(startStates.count))
            criticLosses.append(epochCriticLoss / Float(startStates.count))
        }

        return ImaginationTrainingResult(
            actorLosses: actorLosses,
            criticLosses: criticLosses,
            meanReward: rolloutCount > 0 ? totalReward / (rolloutCount * Float(config.horizon)) : 0,
            meanHorizonLength: rolloutCount > 0 ? totalHorizon / rolloutCount : 0
        )
    }

    // MARK: - Private

    private struct ImaginationRollout {
        let rewards: [MLXArray]     // [horizon] each [batch, 1]
        let values: [MLXArray]      // [horizon+1] each [batch, 1]
        let continues: [MLXArray]   // [horizon] each [batch, 1]
        let rawActions: [MLXArray]  // [horizon] each [batch, 1, driveCount] (pre-tanh)
    }

    private static func rolloutImagination(
        model: ManasMLXCore,
        startState: ManasMLXCoreState,
        config: Config
    ) -> ImaginationRollout {
        var state = startState
        var rewards: [MLXArray] = []
        var values: [MLXArray] = []
        var continues: [MLXArray] = []
        var rawActions: [MLXArray] = []

        // Initial value (uses full features: h + z + adaptZ)
        let initialFeatures = model.imaginationFeatures(from: state)
        values.append(model.predictValue(initialFeatures) ?? MLXArray.zeros([state.fast.dim(0), 1]))

        for _ in 0..<config.horizon {
            // Get action from current policy using full features
            let features = model.imaginationFeatures(from: state)
            let featuresSeq = expandedDimensions(features, axis: 1)
            let rawAction = model.decodeRawDrives(featuresSeq)
            rawActions.append(rawAction)
            var action = tanh(rawAction) * model.config.driveScale
            action = action.squeezed(axis: 1)

            // Step imagination
            let output = model.imaginationStep(action: action, state: state)
            state = output.nextState

            rewards.append(output.rewardPrediction ?? MLXArray.zeros([state.fast.dim(0), 1]))
            // continuePrediction is now raw logits; convert to probability for lambda-returns
            let contLogits = output.continuePrediction ?? MLXArray.ones([state.fast.dim(0), 1])
            continues.append(sigmoid(contLogits))

            // Value at new state (uses full features)
            let newFeatures = model.imaginationFeatures(from: state)
            values.append(model.predictValue(newFeatures) ?? MLXArray.zeros([state.fast.dim(0), 1]))
        }

        return ImaginationRollout(rewards: rewards, values: values, continues: continues, rawActions: rawActions)
    }

    private static func computeLambdaReturns(
        rewards: [MLXArray],
        values: [MLXArray],
        continues: [MLXArray],
        gamma: Float,
        lambda: Float
    ) -> [MLXArray] {
        let horizon = rewards.count
        guard horizon > 0 else { return [] }

        var returns: [MLXArray] = Array(repeating: values[horizon], count: horizon)
        var lastReturn = values[horizon]

        for t in stride(from: horizon - 1, through: 0, by: -1) {
            let nextValue = values[t + 1]
            lastReturn = rewards[t] + gamma * continues[t] * ((1.0 - lambda) * nextValue + lambda * lastReturn)
            returns[t] = lastReturn
        }

        return returns
    }

    private static func stackedMean(_ arrays: [MLXArray]) -> MLXArray {
        guard !arrays.isEmpty else { return MLXArray(Float(0)) }
        var sum = arrays[0].mean()
        for array in arrays.dropFirst() {
            sum = sum + array.mean()
        }
        return sum / Float(arrays.count)
    }

    private static func freezeWorldModel(_ model: ManasMLXCore) {
        model.encoder1.freeze()
        model.encoder2.freeze()
        model.fastGRU.freeze()
        model.slowGRU.freeze()
        model.posteriorNet?.freeze()
        model.priorNet?.freeze()
        model.transitionInputProj?.freeze()
        model.rewardHead1?.freeze()
        model.rewardHead2?.freeze()
        model.continueHead1?.freeze()
        model.continueHead2?.freeze()
        // Nerve network world model modules
        model.sharedEncoder?.freeze()
        model.morphologyEncoder?.freeze()
        model.adaptationModule?.freeze()
        model.typeEmbeddings?.freeze()
        model.descendingEncoder?.freeze()
        model.descendingGate?.freeze()
        // driveHead, sharedActuatorDecoder, and valueHead remain trainable (actor/critic)
    }

    /// Freeze world model and actor (driveHead/sharedActuatorDecoder), leaving only valueHead trainable.
    /// Used during critic training to prevent gradient leak through the actor.
    private static func freezeWorldModelAndActor(_ model: ManasMLXCore) {
        freezeWorldModel(model)
        model.driveHead.freeze()
        model.sharedActuatorDecoder?.freeze()
    }

    private static func unfreezeAll(_ model: ManasMLXCore) {
        model.encoder1.unfreeze()
        model.encoder2.unfreeze()
        model.fastGRU.unfreeze()
        model.slowGRU.unfreeze()
        model.posteriorNet?.unfreeze()
        model.priorNet?.unfreeze()
        model.transitionInputProj?.unfreeze()
        model.rewardHead1?.unfreeze()
        model.rewardHead2?.unfreeze()
        model.continueHead1?.unfreeze()
        model.continueHead2?.unfreeze()
        model.driveHead.unfreeze()
        model.auxHead.unfreeze()
        model.valueHead1?.unfreeze()
        model.valueHead2?.unfreeze()
        // Nerve network modules
        model.sharedEncoder?.unfreeze()
        model.sharedActuatorDecoder?.unfreeze()
        model.morphologyEncoder?.unfreeze()
        model.adaptationModule?.unfreeze()
        model.typeEmbeddings?.unfreeze()
        model.descendingEncoder?.unfreeze()
        model.descendingGate?.unfreeze()
    }

    private static func clipIfNeeded(_ grads: ModuleParameters, maxNorm: Float?) -> ModuleParameters {
        guard let maxNorm else { return grads }
        return clipGradNorm(gradients: grads, maxNorm: maxNorm).0
    }
}
