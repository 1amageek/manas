import Foundation
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
        let lossAndGrad = valueAndGrad(model: model) { model, trunks, targets in
            computeLoss(
                model: model,
                trunks: trunks,
                targets: targets,
                lossConfig: lossConfig,
                descending: nil,
                morphology: nil
            )
        }

        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<epochs {
            var totalLoss: Float = 0
            var totalStepCount = 0
            for batch in batches {
                let (loss, stepCount): (Float, Int) = autoreleasepool {
                    let packedTargets = concatenated(
                        [batch.targetDrives, batch.rewards, batch.continues],
                        axis: -1
                    )
                    let normalizedTargets = ensureBatchTargets(packedTargets)
                    let (lossValue, grads) = gradient(
                        model: model,
                        trunks: batch.trunks,
                        targets: normalizedTargets,
                        lossAndGrad: lossAndGrad,
                        lossConfig: lossConfig,
                        descending: batch.descending,
                        morphology: batch.morphology
                    )
                    let clipped = clipIfNeeded(grads, maxNorm: maxGradNorm)
                    optimizer.update(model: model, gradients: clipped)
                    eval(model, optimizer)
                    return (lossValue.item(Float.self), batchStepCount(normalizedTargets))
                }
                totalLoss += loss * Float(stepCount)
                totalStepCount += stepCount
            }
            epochLosses.append(totalLoss / Float(max(totalStepCount, 1)))
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
        let lossAndGrad = valueAndGrad(model: model) { model, trunks, targets in
            computeLoss(
                model: model,
                trunks: trunks,
                targets: targets,
                lossConfig: lossConfig,
                descending: nil,
                morphology: nil
            )
        }

        var epochLosses: [Float] = []
        model.train(true)

        for epochIndex in 0..<epochs {
            var totalLoss: Float = 0
            let batchCount = max(batches.count, 1)
            var totalStepCount = 0
            for (index, batch) in batches.enumerated() {
                let (loss, stepCount): (Float, Int) = autoreleasepool {
                    let packedTargets = concatenated(
                        [batch.targetDrives, batch.rewards, batch.continues],
                        axis: -1
                    )
                    let normalizedTargets = ensureBatchTargets(packedTargets)
                    let (lossValue, grads) = gradient(
                        model: model,
                        trunks: batch.trunks,
                        targets: normalizedTargets,
                        lossAndGrad: lossAndGrad,
                        lossConfig: lossConfig,
                        descending: batch.descending,
                        morphology: batch.morphology
                    )
                    let clipped = clipIfNeeded(grads, maxNorm: maxGradNorm)
                    optimizer.update(model: model, gradients: clipped)
                    eval(model, optimizer)
                    return (lossValue.item(Float.self), batchStepCount(normalizedTargets))
                }
                totalLoss += loss * Float(stepCount)
                totalStepCount += stepCount
                onProgress?(epochIndex + 1, index + 1, batchCount, loss)
                await Task.yield()
            }
            epochLosses.append(totalLoss / Float(max(totalStepCount, 1)))
            await Task.yield()
        }

        model.train(false)
        return epochLosses
    }

    private static func clipIfNeeded(_ grads: ModuleParameters, maxNorm: Float?) -> ModuleParameters {
        guard let maxNorm else { return grads }
        return clipGradNorm(gradients: grads, maxNorm: maxNorm).0
    }

    private static func gradient(
        model: ManasMLXCore,
        trunks: MLXArray,
        targets: MLXArray,
        lossAndGrad: (ManasMLXCore, MLXArray, MLXArray) -> (MLXArray, ModuleParameters),
        lossConfig: WorldModelLoss.Config,
        descending: MLXArray?,
        morphology: MLXArray?
    ) -> (MLXArray, ModuleParameters) {
        guard descending != nil || morphology != nil else {
            return lossAndGrad(model, trunks, targets)
        }

        let contextualLossAndGrad = valueAndGrad(model: model) { model, trunks, targets in
            computeLoss(
                model: model,
                trunks: trunks,
                targets: targets,
                lossConfig: lossConfig,
                descending: descending,
                morphology: morphology
            )
        }
        return contextualLossAndGrad(model, trunks, targets)
    }

    private static func computeLoss(
        model: ManasMLXCore,
        trunks: MLXArray,
        targets: MLXArray,
        lossConfig: WorldModelLoss.Config,
        descending: MLXArray?,
        morphology: MLXArray?
    ) -> MLXArray {
        let rssmOutput = model.forwardRSSM(
            trunks: trunks,
            descending: descending,
            morphology: morphology
        )
        let driveCount = model.config.driveCount
        let driveTargets = targets[.ellipsis, 0..<driveCount]
        let rewards = targets[.ellipsis, driveCount..<(driveCount + 1)]
        let continues = targets[.ellipsis, (driveCount + 1)..<(driveCount + 2)]

        let batch = ManasMLXWorldModelBatch(
            trunks: trunks,
            targetDrives: driveTargets,
            rewards: rewards,
            continues: continues
        )
        return WorldModelLoss.compute(rssmOutput: rssmOutput, batch: batch, config: lossConfig)
    }

    private static func ensureBatchTargets(_ targets: MLXArray) -> MLXArray {
        let shape = targets.shape
        guard shape.count == 2 else { return targets }
        return targets.reshaped([1, shape[0], shape[1]])
    }

    private static func batchStepCount(_ targets: MLXArray) -> Int {
        let shape = targets.shape
        guard shape.count >= 3 else {
            return shape.first ?? 1
        }
        return max(1, shape[0] * shape[1])
    }
}
