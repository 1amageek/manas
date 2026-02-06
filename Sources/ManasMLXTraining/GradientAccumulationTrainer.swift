import MLX
import MLXNN
import MLXOptimizers
import ManasMLXModels

public enum GradientAccumulationTrainer {
    public static func trainCore(
        model: ManasMLXCore,
        batches: [ManasMLXSequenceBatch],
        config: ManasMLXGradientAccumulationConfig,
        epochs: Int
    ) -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, trunks, targets in
            let output = model.forward(trunks: trunks)
            let loss = mseLoss(predictions: output.drives, targets: targets, reduction: .mean)
            return loss * config.driveLossWeight
        }

        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<epochs {
            var totalLoss: Float = 0
            var accumulatedGrads: ModuleParameters?
            var stepCount = 0

            for (batchIndex, batch) in batches.enumerated() {
                let targets = ensureBatchTargets(batch.targetDrives)
                let (lossValue, grads) = lg(model, batch.trunks, targets)
                totalLoss += lossValue.item(Float.self)

                if accumulatedGrads == nil {
                    accumulatedGrads = grads
                } else {
                    accumulatedGrads = addGradients(accumulatedGrads!, grads)
                }
                stepCount += 1

                if stepCount >= config.accumulationSteps || batchIndex == batches.count - 1 {
                    if var accumulated = accumulatedGrads {
                        accumulated = scaleGradients(accumulated, scale: 1.0 / Float(stepCount))
                        let clipped = clipIfNeeded(accumulated, maxNorm: config.maxGradNorm)
                        optimizer.update(model: model, gradients: clipped)
                        eval(model, optimizer)
                    }
                    accumulatedGrads = nil
                    stepCount = 0
                }
            }

            epochLosses.append(totalLoss / Float(max(batches.count, 1)))
        }

        model.train(false)
        return epochLosses
    }

    @MainActor
    public static func trainCoreAsync(
        model: ManasMLXCore,
        batches: [ManasMLXSequenceBatch],
        config: ManasMLXGradientAccumulationConfig,
        epochs: Int,
        onProgress: (@Sendable (_ epoch: Int, _ step: Int, _ loss: Float) -> Void)? = nil
    ) async -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, trunks, targets in
            let output = model.forward(trunks: trunks)
            let loss = mseLoss(predictions: output.drives, targets: targets, reduction: .mean)
            return loss * config.driveLossWeight
        }

        var epochLosses: [Float] = []
        model.train(true)

        for epochIndex in 0..<epochs {
            var totalLoss: Float = 0
            var accumulatedGrads: ModuleParameters?
            var stepCount = 0
            var updateCount = 0

            for (batchIndex, batch) in batches.enumerated() {
                let targets = ensureBatchTargets(batch.targetDrives)
                let (lossValue, grads) = lg(model, batch.trunks, targets)
                let loss = lossValue.item(Float.self)
                totalLoss += loss

                if accumulatedGrads == nil {
                    accumulatedGrads = grads
                } else {
                    accumulatedGrads = addGradients(accumulatedGrads!, grads)
                }
                stepCount += 1

                if stepCount >= config.accumulationSteps || batchIndex == batches.count - 1 {
                    if var accumulated = accumulatedGrads {
                        accumulated = scaleGradients(accumulated, scale: 1.0 / Float(stepCount))
                        let clipped = clipIfNeeded(accumulated, maxNorm: config.maxGradNorm)
                        optimizer.update(model: model, gradients: clipped)
                        eval(model, optimizer)
                    }
                    updateCount += 1
                    onProgress?(epochIndex + 1, updateCount, loss)
                    accumulatedGrads = nil
                    stepCount = 0
                    await Task.yield()
                }
            }

            epochLosses.append(totalLoss / Float(max(batches.count, 1)))
            await Task.yield()
        }

        model.train(false)
        return epochLosses
    }

    private static func addGradients(_ a: ModuleParameters, _ b: ModuleParameters) -> ModuleParameters {
        a.mapValues(b) { aItem, bItem in
            guard let bArr = bItem else {
                return aItem
            }
            return aItem + bArr
        }
    }

    private static func scaleGradients(_ grads: ModuleParameters, scale: Float) -> ModuleParameters {
        grads.mapValues { $0 * scale }
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
