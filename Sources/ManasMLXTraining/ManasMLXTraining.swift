import MLX
import MLXNN
import MLXOptimizers
import ManasMLXModels

public struct ManasMLXTrainingConfig: Sendable, Equatable {
    public let epochs: Int
    public let learningRate: Float
    public let maxGradNorm: Float?
    public let driveLossWeight: Float
    public let auxLossWeight: Float

    public init(
        epochs: Int,
        learningRate: Float,
        maxGradNorm: Float? = 1.0,
        driveLossWeight: Float = 1.0,
        auxLossWeight: Float = 0.1
    ) {
        self.epochs = epochs
        self.learningRate = learningRate
        self.maxGradNorm = maxGradNorm
        self.driveLossWeight = driveLossWeight
        self.auxLossWeight = auxLossWeight
    }
}

public struct ManasMLXSequenceBatch {
    public let trunks: MLXArray
    public let targetDrives: MLXArray

    public init(trunks: MLXArray, targetDrives: MLXArray) {
        self.trunks = trunks
        self.targetDrives = targetDrives
    }
}

public struct ManasMLXAuxSequenceBatch {
    public let trunks: MLXArray
    public let targetDrives: MLXArray
    public let targetAux: MLXArray

    public init(trunks: MLXArray, targetDrives: MLXArray, targetAux: MLXArray) {
        self.trunks = trunks
        self.targetDrives = targetDrives
        self.targetAux = targetAux
    }
}

public struct ManasMLXReflexBatch {
    public let inputs: MLXArray
    public let targetClamp: MLXArray
    public let targetDamping: MLXArray
    public let targetDelta: MLXArray

    public init(inputs: MLXArray, targetClamp: MLXArray, targetDamping: MLXArray, targetDelta: MLXArray) {
        self.inputs = inputs
        self.targetClamp = targetClamp
        self.targetDamping = targetDamping
        self.targetDelta = targetDelta
    }
}

public struct ManasMLXNerveSequenceBatch {
    public let ascendingValues: MLXArray
    public let ascendingTypeIndices: MLXArray
    public let descendingValues: MLXArray?
    public let descendingTypeIndices: MLXArray?
    public let morphology: MLXArray?
    public let targetCommands: MLXArray
    public let actuatorTypeIndices: MLXArray

    public init(
        ascendingValues: MLXArray,
        ascendingTypeIndices: MLXArray,
        descendingValues: MLXArray? = nil,
        descendingTypeIndices: MLXArray? = nil,
        morphology: MLXArray? = nil,
        targetCommands: MLXArray,
        actuatorTypeIndices: MLXArray
    ) {
        self.ascendingValues = ascendingValues
        self.ascendingTypeIndices = ascendingTypeIndices
        self.descendingValues = descendingValues
        self.descendingTypeIndices = descendingTypeIndices
        self.morphology = morphology
        self.targetCommands = targetCommands
        self.actuatorTypeIndices = actuatorTypeIndices
    }
}

public enum ManasMLXTrainer {
    public enum TrainingError: Error, Equatable {
        case auxDisabled
    }

    public static func trainCoreSupervised(
        model: ManasMLXCore,
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
    public static func trainCoreSupervisedAsync(
        model: ManasMLXCore,
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

    public static func trainCoreWithAux(
        model: ManasMLXCore,
        batches: [ManasMLXAuxSequenceBatch],
        config: ManasMLXTrainingConfig
    ) throws -> [Float] {
        guard model.config.auxEnabled else {
            throw TrainingError.auxDisabled
        }
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, trunks, targets in
            let output = model.forward(trunks: trunks)
            let driveCount = model.config.driveCount
            let auxSize = model.config.auxSize
            let driveTargets = targets[.ellipsis, 0..<driveCount]
            let auxTargets = targets[.ellipsis, driveCount..<(driveCount + auxSize)]
            let driveLoss = mseLoss(predictions: output.drives, targets: driveTargets, reduction: .mean)
            let auxPred = output.aux ?? output.drives
            let auxLoss = mseLoss(predictions: auxPred, targets: auxTargets, reduction: .mean)
            return driveLoss * config.driveLossWeight + auxLoss * config.auxLossWeight
        }

        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<config.epochs {
            var totalLoss: Float = 0
            for batch in batches {
                let combinedTargets = makeAuxTargets(
                    driveTargets: batch.targetDrives,
                    auxTargets: batch.targetAux
                )
                let (lossValue, grads) = lg(model, batch.trunks, combinedTargets)
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
    public static func trainCoreWithAuxAsync(
        model: ManasMLXCore,
        batches: [ManasMLXAuxSequenceBatch],
        config: ManasMLXTrainingConfig,
        onProgress: (@Sendable (_ epoch: Int, _ batch: Int, _ batchCount: Int, _ loss: Float) -> Void)? = nil
    ) async throws -> [Float] {
        guard model.config.auxEnabled else {
            throw TrainingError.auxDisabled
        }
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, trunks, targets in
            let output = model.forward(trunks: trunks)
            let driveCount = model.config.driveCount
            let auxSize = model.config.auxSize
            let driveTargets = targets[.ellipsis, 0..<driveCount]
            let auxTargets = targets[.ellipsis, driveCount..<(driveCount + auxSize)]
            let driveLoss = mseLoss(predictions: output.drives, targets: driveTargets, reduction: .mean)
            let auxPred = output.aux ?? output.drives
            let auxLoss = mseLoss(predictions: auxPred, targets: auxTargets, reduction: .mean)
            return driveLoss * config.driveLossWeight + auxLoss * config.auxLossWeight
        }

        var epochLosses: [Float] = []
        model.train(true)

        for epochIndex in 0..<config.epochs {
            var totalLoss: Float = 0
            let batchCount = max(batches.count, 1)
            for (index, batch) in batches.enumerated() {
                let combinedTargets = makeAuxTargets(
                    driveTargets: batch.targetDrives,
                    auxTargets: batch.targetAux
                )
                let (lossValue, grads) = lg(model, batch.trunks, combinedTargets)
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

    public static func trainReflexSupervised(
        model: ManasMLXReflex,
        batches: [ManasMLXReflexBatch],
        config: ManasMLXTrainingConfig
    ) -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, inputs, targets in
            let output = model.forward(inputs)
            let driveCount = model.config.driveCount
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

    @MainActor
    public static func trainReflexSupervisedAsync(
        model: ManasMLXReflex,
        batches: [ManasMLXReflexBatch],
        config: ManasMLXTrainingConfig
    ) async -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)
        let lg = valueAndGrad(model: model) { model, inputs, targets in
            let output = model.forward(inputs)
            let driveCount = model.config.driveCount
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
                await Task.yield()
            }
            epochLosses.append(totalLoss / Float(max(batches.count, 1)))
            await Task.yield()
        }

        model.train(false)
        return epochLosses
    }

    public static func trainNerveCoreSupervised(
        model: ManasMLXCore,
        batches: [ManasMLXNerveSequenceBatch],
        config: ManasMLXTrainingConfig
    ) -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)

        var epochLosses: [Float] = []
        model.train(true)

        for _ in 0..<config.epochs {
            var totalLoss: Float = 0
            for batch in batches {
                let targets = ensureBatchTargets(batch.targetCommands)
                let desc = batch.descendingValues
                let morph = batch.morphology
                let (lossValue, grads) = valueAndGrad(model: model) { model, trunks, targets in
                    let output = model.forward(trunks: trunks, descending: desc, morphology: morph)
                    let loss = mseLoss(predictions: output.drives, targets: targets, reduction: .mean)
                    return loss * config.driveLossWeight
                }(model, batch.ascendingValues, targets)
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
    public static func trainNerveCoreSupervisedAsync(
        model: ManasMLXCore,
        batches: [ManasMLXNerveSequenceBatch],
        config: ManasMLXTrainingConfig,
        onProgress: (@Sendable (_ epoch: Int, _ batch: Int, _ batchCount: Int, _ loss: Float) -> Void)? = nil
    ) async -> [Float] {
        let optimizer = Adam(learningRate: config.learningRate)

        var epochLosses: [Float] = []
        model.train(true)

        for epochIndex in 0..<config.epochs {
            var totalLoss: Float = 0
            let batchCount = max(batches.count, 1)
            for (index, batch) in batches.enumerated() {
                let targets = ensureBatchTargets(batch.targetCommands)
                let desc = batch.descendingValues
                let morph = batch.morphology
                let (lossValue, grads) = valueAndGrad(model: model) { model, trunks, targets in
                    let output = model.forward(trunks: trunks, descending: desc, morphology: morph)
                    let loss = mseLoss(predictions: output.drives, targets: targets, reduction: .mean)
                    return loss * config.driveLossWeight
                }(model, batch.ascendingValues, targets)
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

    private static func clipIfNeeded(_ grads: ModuleParameters, maxNorm: Float?) -> ModuleParameters {
        guard let maxNorm else { return grads }
        return clipGradNorm(gradients: grads, maxNorm: maxNorm).0
    }

    private static func makeAuxTargets(
        driveTargets: MLXArray,
        auxTargets: MLXArray
    ) -> MLXArray {
        let driveShape = driveTargets.shape
        guard driveShape.count == 2 else {
            return concatenated([driveTargets, auxTargets], axis: -1)
        }
        let sequenceLength = driveShape[0]
        let driveCount = driveShape[1]
        let auxSize = auxTargets.shape.last ?? 0
        let expandedAux = repeated(auxTargets, count: sequenceLength, axis: 0)
            .reshaped([sequenceLength, auxSize])
        let driveExpanded = driveTargets.reshaped([1, sequenceLength, driveCount])
        let auxExpanded = expandedAux.reshaped([1, sequenceLength, auxSize])
        return concatenated([driveExpanded, auxExpanded], axis: -1)
    }

    private static func ensureBatchTargets(_ targets: MLXArray) -> MLXArray {
        let shape = targets.shape
        guard shape.count == 2 else { return targets }
        return targets.reshaped([1, shape[0], shape[1]])
    }
}
