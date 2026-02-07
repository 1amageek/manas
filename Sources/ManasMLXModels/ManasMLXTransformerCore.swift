import MLX
import MLXNN

public final class ManasMLXTransformerCore: Module {
    public let config: ManasMLXTransformerCoreConfig

    @ModuleInfo public var inputProjection: Linear
    @ModuleInfo public var positionalEncoding: SinusoidalPositionalEncoding
    @ModuleInfo public var blocks: [ManasTransformerBlock]
    @ModuleInfo public var outputNorm: LayerNorm
    @ModuleInfo public var driveHead: Linear
    @ModuleInfo public var auxHead: Linear

    public init(config: ManasMLXTransformerCoreConfig) {
        self.config = config
        self._inputProjection.wrappedValue = Linear(config.inputSize, config.dModel)
        self._positionalEncoding.wrappedValue = SinusoidalPositionalEncoding(dimensions: config.dModel)
        self._blocks.wrappedValue = (0..<config.numLayers).map { _ in
            ManasTransformerBlock(
                dModel: config.dModel,
                numHeads: config.numHeads,
                ffnHiddenSize: config.ffnHiddenSize,
                goalSize: config.goalSize
            )
        }
        self._outputNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._driveHead.wrappedValue = Linear(config.dModel, config.driveCount)
        self._auxHead.wrappedValue = Linear(config.dModel, max(config.auxSize, 1))
    }

    public func forward(trunks: MLXArray, goals: MLXArray? = nil) -> ManasMLXTransformerCoreOutput {
        let sequence = normalizeSequence(trunks)
        var h = inputProjection(sequence)

        let seqLen = h.dim(1)
        let positions = MLXArray(0..<seqLen).asType(.float32)
        let pe = positionalEncoding(positions)
        h = h + pe

        let mask: MLXArray?
        if config.useCausalMask {
            mask = MultiHeadAttention.createAdditiveCausalMask(seqLen)
        } else {
            mask = nil
        }

        for block in blocks {
            h = block(h, goals: goals, mask: mask)
        }

        h = outputNorm(h)

        var drives = driveHead(h)
        drives = tanh(drives) * config.driveScale

        let auxOutput: MLXArray?
        if config.auxEnabled {
            auxOutput = sanitizeAux(auxHead(h))
        } else {
            auxOutput = nil
        }

        return ManasMLXTransformerCoreOutput(drives: drives, aux: auxOutput)
    }

    public func callAsFunction(_ trunks: MLXArray) -> MLXArray {
        forward(trunks: trunks, goals: nil).drives
    }

    private func normalizeSequence(_ input: MLXArray) -> MLXArray {
        MLXModelUtils.normalizeSequence(input)
    }

    private func sanitizeAux(_ aux: MLXArray) -> MLXArray? {
        MLXModelUtils.sanitizeAux(aux, expectedSize: config.auxSize)
    }
}

public struct ManasMLXTransformerCoreOutput {
    public let drives: MLXArray
    public let aux: MLXArray?

    public init(drives: MLXArray, aux: MLXArray?) {
        self.drives = drives
        self.aux = aux
    }
}
