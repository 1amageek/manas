import MLX
import MLXNN
import MLXOptimizers
import ManasMLXModels

public enum ManasMLXLoRATrainer {
    public static func trainLoRACore(
        model: ManasMLXLoRACore,
        batches: [ManasMLXSequenceBatch],
        config: ManasMLXTrainingConfig
    ) -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, trunks, targets in
            let output = model.forward(trunks: trunks)
            let loss = mseLoss(predictions: output.drives, targets: targets, reduction: .mean)
            return loss * config.driveLossWeight
        }

        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<config.epochs {
            var totalLoss: Float = 0
            for batch in batches {
                let targets = ensureBatchTargets(batch.targetDrives)
                let (lossValue, grads) = lg(model, batch.trunks, targets)
                let clipped = clipIfNeeded(grads, maxNorm: config.maxGradNorm)
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
    public static func trainLORACoreAsync(
        model: ManasMLXLoRACore,
        batches: [ManasMLXSequenceBatch],
        config: ManasMLXTrainingConfig,
        onProgress: (@Sendable (_ epoch: Int, _ batch: Int, _ batchCount: Int, _ loss: Float) -> Void)? = nil
    ) async -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, trunks, targets in
            let output = model.forward(trunks: trunks)
            let loss = mseLoss(predictions: output.drives, targets: targets, reduction: .mean)
            return loss * config.driveLossWeight
        }

        var epochLosses: [Float] = []
        model.train(true)

        for epochIndex in 0..<config.epochs {
            var totalLoss: Float = 0
            let batchCount = max(batches.count, 1)
            for (index, batch) in batches.enumerated() {
                let targets = ensureBatchTargets(batch.targetDrives)
                let (lossValue, grads) = lg(model, batch.trunks, targets)
                let clipped = clipIfNeeded(grads, maxNorm: config.maxGradNorm)
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

    public static func trainLoRAReflex(
        model: ManasMLXLoRAReflex,
        batches: [ManasMLXReflexBatch],
        config: ManasMLXTrainingConfig
    ) -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, inputs, targets in
            let output = model.forward(inputs)
            let driveCount = model.reflexConfig.driveCount
            let clampTargets = targets[.ellipsis, 0..<driveCount]
            let dampingTargets = targets[.ellipsis, driveCount..<(driveCount * 2)]
            let deltaTargets = targets[.ellipsis, (driveCount * 2)..<(driveCount * 3)]
            let clampLoss = mseLoss(predictions: output.clamp, targets: clampTargets, reduction: .mean)
            let dampingLoss = mseLoss(predictions: output.damping, targets: dampingTargets, reduction: .mean)
            let deltaLoss = mseLoss(predictions: output.delta, targets: deltaTargets, reduction: .mean)
            return (clampLoss + dampingLoss + deltaLoss) * config.driveLossWeight
        }

        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<config.epochs {
            var totalLoss: Float = 0
            for batch in batches {
                let combinedTargets = concatenated(
                    [batch.targetClamp, batch.targetDamping, batch.targetDelta],
                    axis: -1
                )
                let (lossValue, grads) = lg(model, batch.inputs, combinedTargets)
                let clipped = clipIfNeeded(grads, maxNorm: config.maxGradNorm)
                optimizer.update(model: model, gradients: clipped)
                eval(model, optimizer)
                totalLoss += lossValue.item(Float.self)
            }
            epochLosses.append(totalLoss / Float(max(batches.count, 1)))
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
