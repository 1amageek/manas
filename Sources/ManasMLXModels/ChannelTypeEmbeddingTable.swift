import MLX
import MLXNN

public final class ChannelTypeEmbeddingTable: Module {
    @ModuleInfo public var embedding: Embedding

    public let channelCount: Int
    public let embeddingDim: Int

    public init(channelCount: Int, embeddingDim: Int) {
        self.channelCount = channelCount
        self.embeddingDim = embeddingDim
        self._embedding.wrappedValue = Embedding(embeddingCount: channelCount, dimensions: embeddingDim)
    }

    public func callAsFunction(_ indices: MLXArray) -> MLXArray {
        embedding(indices)
    }
}
