import MLX
import MLXNN
import MLXOptimizers
import ManasMLXModels

/// Trains the world model (encoder, GRUs, posterior/prior, prediction heads)
/// using the combined world model loss.
public enum WorldModelTrainer {

    public static func train(
        model: ManasMLXCore,
        batches: [ManasMLXWorldModelBatch],
        lossConfig: WorldModelLoss.Config,
        learningRate: Float = 0.001,
        maxGradNorm: Float? = 1.0,
        epochs: Int
    ) -> [Float] {
        let optimizer = Adam(learningRate: learningRate)

        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<epochs {
            var totalLoss: Float = 0
            for batch in batches {
                let desc = batch.descending
                let morph = batch.morphology
                let lg = valueAndGrad(model: model) { model, trunks, targets in
                    let rssmOutput = model.forwardRSSM(
                        trunks: trunks, descending: desc, morphology: morph
                    )
                    let driveCount = model.config.driveCount
                    let driveTargets = targets[.ellipsis, 0..<driveCount]
                    let rewards = targets[.ellipsis, driveCount..<(driveCount + 1)]
                    let continues = targets[.ellipsis, (driveCount + 1)..<(driveCount + 2)]

                    let b = ManasMLXWorldModelBatch(
                        trunks: trunks,
                        targetDrives: driveTargets,
                        rewards: rewards,
                        continues: continues
                    )
                    return WorldModelLoss.compute(rssmOutput: rssmOutput, batch: b, config: lossConfig)
                }
                // Pack targets for valueAndGrad closure
                let packedTargets = concatenated(
                    [batch.targetDrives, batch.rewards, batch.continues],
                    axis: -1
                )
                let normalizedTargets = ensureBatchTargets(packedTargets)
                let (lossValue, grads) = lg(model, batch.trunks, normalizedTargets)
                let clipped = clipIfNeeded(grads, maxNorm: maxGradNorm)
                optimizer.update(model: model, gradients: clipped)
                eval(model, optimizer)
                totalLoss += lossValue.item(Float.self)
            }
            epochLosses.append(totalLoss / Float(max(batches.count, 1)))
        }

        model.train(false)
        return epochLosses
    }

    @MainActor
    public static func trainAsync(
        model: ManasMLXCore,
        batches: [ManasMLXWorldModelBatch],
        lossConfig: WorldModelLoss.Config,
        learningRate: Float = 0.001,
        maxGradNorm: Float? = 1.0,
        epochs: Int,
        onProgress: (@Sendable (_ epoch: Int, _ batch: Int, _ batchCount: Int, _ loss: Float) -> Void)? = nil
    ) async -> [Float] {
        let optimizer = Adam(learningRate: learningRate)

        var epochLosses: [Float] = []
        model.train(true)

        for epochIndex in 0..<epochs {
            var totalLoss: Float = 0
            let batchCount = max(batches.count, 1)
            for (index, batch) in batches.enumerated() {
                let desc = batch.descending
                let morph = batch.morphology
                let lg = valueAndGrad(model: model) { model, trunks, targets in
                    let rssmOutput = model.forwardRSSM(
                        trunks: trunks, descending: desc, morphology: morph
                    )
                    let driveCount = model.config.driveCount
                    let driveTargets = targets[.ellipsis, 0..<driveCount]
                    let rewards = targets[.ellipsis, driveCount..<(driveCount + 1)]
                    let continues = targets[.ellipsis, (driveCount + 1)..<(driveCount + 2)]

                    let b = ManasMLXWorldModelBatch(
                        trunks: trunks,
                        targetDrives: driveTargets,
                        rewards: rewards,
                        continues: continues
                    )
                    return WorldModelLoss.compute(rssmOutput: rssmOutput, batch: b, config: lossConfig)
                }
                let packedTargets = concatenated(
                    [batch.targetDrives, batch.rewards, batch.continues],
                    axis: -1
                )
                let normalizedTargets = ensureBatchTargets(packedTargets)
                let (lossValue, grads) = lg(model, batch.trunks, normalizedTargets)
                let clipped = clipIfNeeded(grads, maxNorm: maxGradNorm)
                optimizer.update(model: model, gradients: clipped)
                eval(model, optimizer)
                let loss = lossValue.item(Float.self)
                totalLoss += loss
                onProgress?(epochIndex + 1, index + 1, batchCount, loss)
                await Task.yield()
            }
            epochLosses.append(totalLoss / Float(batchCount))
            await Task.yield()
        }

        model.train(false)
        return epochLosses
    }

    private static func clipIfNeeded(_ grads: ModuleParameters, maxNorm: Float?) -> ModuleParameters {
        guard let maxNorm else { return grads }
        return clipGradNorm(gradients: grads, maxNorm: maxNorm).0
    }

    private static func ensureBatchTargets(_ targets: MLXArray) -> MLXArray {
        let shape = targets.shape
        guard shape.count == 2 else { return targets }
        return targets.reshaped([1, shape[0], shape[1]])
    }
}
