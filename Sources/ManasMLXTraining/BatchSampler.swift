import MLX
import ManasMLXModels

/// Sampling strategies for creating training batches from a dataset.
public enum BatchSampler {

    public enum Strategy: Sendable, Equatable {
        case sequential
        case shuffled(seed: UInt64)
    }

    /// Create sequence batches for Core training from flat arrays.
    ///
    /// - Parameters:
    ///   - inputs: All input vectors (trunk data) as a flat MLXArray [totalSamples, inputSize]
    ///   - targets: All target vectors (drive data) as a flat MLXArray [totalSamples, driveCount]
    ///   - batchSize: Number of sequences per batch
    ///   - sequenceLength: Number of timesteps per sequence
    ///   - strategy: Sampling strategy
    /// - Returns: Array of ManasMLXSequenceBatch
    public static func createCoreBatches(
        inputs: MLXArray,
        targets: MLXArray,
        batchSize: Int,
        sequenceLength: Int,
        strategy: Strategy = .sequential
    ) -> [ManasMLXSequenceBatch] {
        let totalSamples = inputs.dim(0)
        let totalSequences = totalSamples / sequenceLength
        guard totalSequences > 0 else { return [] }

        let indices = orderedIndices(count: totalSequences, strategy: strategy)
        let totalBatches = totalSequences / batchSize

        return (0..<totalBatches).map { batchIdx in
            let batchStart = batchIdx * batchSize
            let batchEnd = min(batchStart + batchSize, totalSequences)

            var inputSequences: [MLXArray] = []
            var targetSequences: [MLXArray] = []

            for seqIdx in batchStart..<batchEnd {
                let i = indices[seqIdx]
                let start = i * sequenceLength
                let end = start + sequenceLength
                inputSequences.append(inputs[start..<end])
                targetSequences.append(targets[start..<end])
            }

            let batchInputs = stacked(inputSequences)
            let batchTargets = stacked(targetSequences)
            return ManasMLXSequenceBatch(trunks: batchInputs, targetDrives: batchTargets)
        }
    }

    /// Create reflex batches from flat arrays.
    ///
    /// - Parameters:
    ///   - inputs: Input vectors [totalSamples, inputSize]
    ///   - targetClamp: Clamp targets [totalSamples, driveCount]
    ///   - targetDamping: Damping targets [totalSamples, driveCount]
    ///   - targetDelta: Delta targets [totalSamples, driveCount]
    ///   - batchSize: Samples per batch
    ///   - strategy: Sampling strategy
    /// - Returns: Array of ManasMLXReflexBatch
    public static func createReflexBatches(
        inputs: MLXArray,
        targetClamp: MLXArray,
        targetDamping: MLXArray,
        targetDelta: MLXArray,
        batchSize: Int,
        strategy: Strategy = .sequential
    ) -> [ManasMLXReflexBatch] {
        let totalSamples = inputs.dim(0)
        guard totalSamples > 0 else { return [] }

        let indices = orderedIndices(count: totalSamples, strategy: strategy)
        let totalBatches = (totalSamples + batchSize - 1) / batchSize

        return (0..<totalBatches).map { batchIdx in
            let batchStart = batchIdx * batchSize
            let batchEnd = min(batchStart + batchSize, totalSamples)

            var batchIndices: [Int] = []
            for i in batchStart..<batchEnd {
                batchIndices.append(indices[i])
            }

            let indexArray = MLXArray(batchIndices)
            return ManasMLXReflexBatch(
                inputs: inputs[indexArray],
                targetClamp: targetClamp[indexArray],
                targetDamping: targetDamping[indexArray],
                targetDelta: targetDelta[indexArray]
            )
        }
    }

    // MARK: - Private

    private static func orderedIndices(count: Int, strategy: Strategy) -> [Int] {
        var indices = Array(0..<count)
        switch strategy {
        case .sequential:
            break
        case .shuffled(let seed):
            var rng = SplitMix64Sampler(seed: seed)
            for i in stride(from: count - 1, through: 1, by: -1) {
                let j = Int(rng.next() % UInt64(i + 1))
                indices.swapAt(i, j)
            }
        }
        return indices
    }
}

/// Simple PRNG for deterministic shuffling within ManasMLXTraining.
private struct SplitMix64Sampler {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
