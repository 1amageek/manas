import MLX
import MLXNN

public final class ManasTransformerBlock: Module {
    @ModuleInfo public var selfAttention: MultiHeadAttention
    @ModuleInfo public var selfAttentionNorm: LayerNorm
    @ModuleInfo public var crossAttention: MultiHeadAttention?
    @ModuleInfo public var crossAttentionNorm: LayerNorm?
    @ModuleInfo public var ffn: ManasFFN
    @ModuleInfo public var ffnNorm: LayerNorm

    public init(dModel: Int, numHeads: Int, ffnHiddenSize: Int, goalSize: Int? = nil) {
        self._selfAttention.wrappedValue = MultiHeadAttention(dimensions: dModel, numHeads: numHeads)
        self._selfAttentionNorm.wrappedValue = LayerNorm(dimensions: dModel)
        if let goalSize {
            self._crossAttention.wrappedValue = MultiHeadAttention(
                dimensions: dModel,
                numHeads: numHeads,
                keyInputDimensions: goalSize,
                valueInputDimensions: goalSize
            )
            self._crossAttentionNorm.wrappedValue = LayerNorm(dimensions: dModel)
        } else {
            self._crossAttention.wrappedValue = nil
            self._crossAttentionNorm.wrappedValue = nil
        }
        self._ffn.wrappedValue = ManasFFN(dModel: dModel, hiddenSize: ffnHiddenSize)
        self._ffnNorm.wrappedValue = LayerNorm(dimensions: dModel)
    }

    public func callAsFunction(_ x: MLXArray, goals: MLXArray? = nil, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        let normed = selfAttentionNorm(h)
        h = h + selfAttention(normed, keys: normed, values: normed, mask: mask)

        if let crossAttention, let crossAttentionNorm, let goals {
            let crossNormed = crossAttentionNorm(h)
            h = h + crossAttention(crossNormed, keys: goals, values: goals)
        }

        let ffnNormed = ffnNorm(h)
        h = h + ffn(ffnNormed)
        return h
    }
}
