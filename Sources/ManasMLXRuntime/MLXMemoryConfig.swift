/// Unified memory configuration for MLX inference and training.
///
/// Controls memory optimization strategies such as KV caching for Transformer
/// inference and gradient checkpointing for memory-efficient training.
public struct MLXMemoryConfig: Sendable, Equatable {
    /// Whether to use gradient checkpointing during training.
    /// Trades compute for memory by recomputing activations during backward pass.
    public let enableGradientCheckpointing: Bool

    /// Whether to use KV caching during Transformer inference.
    public let kvCacheEnabled: Bool

    /// Maximum sequence length for the KV cache.
    public let kvCacheMaxLength: Int

    /// Maximum number of GRU states to retain for warmup.
    public let gruStateCacheEnabled: Bool

    public init(
        enableGradientCheckpointing: Bool = false,
        kvCacheEnabled: Bool = true,
        kvCacheMaxLength: Int = 512,
        gruStateCacheEnabled: Bool = true
    ) {
        self.enableGradientCheckpointing = enableGradientCheckpointing
        self.kvCacheEnabled = kvCacheEnabled
        self.kvCacheMaxLength = kvCacheMaxLength
        self.gruStateCacheEnabled = gruStateCacheEnabled
    }
}
