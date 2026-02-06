import MLX

/// Key-Value cache for Transformer inference, avoiding recomputation of
/// attention keys and values for previously seen sequence positions.
public struct KVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public let maxLength: Int

    public init(maxLength: Int) {
        self.keys = nil
        self.values = nil
        self.maxLength = maxLength
    }

    /// Append new key/value tensors and return the full cached tensors.
    ///
    /// - Parameters:
    ///   - key: New key tensor [batch, newSeqLen, dims]
    ///   - value: New value tensor [batch, newSeqLen, dims]
    /// - Returns: Tuple of (allKeys, allValues) including cached history.
    public mutating func append(key: MLXArray, value: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            keys = concatenated([existingKeys, key], axis: 1)
            values = concatenated([existingValues, value], axis: 1)
        } else {
            keys = key
            values = value
        }

        // Trim to maxLength if exceeded
        if let k = keys, k.dim(1) > maxLength {
            let start = k.dim(1) - maxLength
            keys = k[0..., start..<k.dim(1)]
            values = values![0..., start..<values!.dim(1)]
        }

        return (keys!, values!)
    }

    /// Current sequence length stored in the cache.
    public var currentLength: Int {
        keys?.dim(1) ?? 0
    }

    /// Reset the cache, discarding all stored keys and values.
    public mutating func reset() {
        keys = nil
        values = nil
    }
}
