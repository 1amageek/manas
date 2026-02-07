import MLX
import MLXNN

@available(*, deprecated, message: "Use ManasMLXCore with descendingSize > 0 instead")
public final class ManasMLXGoalCore: Module {
    public let config: ManasMLXGoalCoreConfig

    @ModuleInfo public var encoder1: Linear
    @ModuleInfo public var encoder2: Linear
    @ModuleInfo public var goalEncoder: Linear
    @ModuleInfo public var goalGate: Linear
    @ModuleInfo public var fastGRU: GRU
    @ModuleInfo public var slowGRU: GRU
    @ModuleInfo public var driveHead: Linear
    @ModuleInfo public var auxHead: Linear

    public init(config: ManasMLXGoalCoreConfig) {
        self.config = config
        let combinedInput = config.embeddingSize + config.goalEmbeddingSize
        self._encoder1.wrappedValue = Linear(config.inputSize, config.embeddingSize)
        self._encoder2.wrappedValue = Linear(config.embeddingSize, config.embeddingSize)
        self._goalEncoder.wrappedValue = Linear(config.goalSize, config.goalEmbeddingSize)
        self._goalGate.wrappedValue = Linear(config.goalEmbeddingSize, config.goalEmbeddingSize)
        self._fastGRU.wrappedValue = GRU(inputSize: combinedInput, hiddenSize: config.fastHiddenSize)
        self._slowGRU.wrappedValue = GRU(inputSize: combinedInput, hiddenSize: config.slowHiddenSize)
        self._driveHead.wrappedValue = Linear(config.fastHiddenSize + config.slowHiddenSize, config.driveCount)
        self._auxHead.wrappedValue = Linear(config.fastHiddenSize + config.slowHiddenSize, max(config.auxSize, 1))
    }

    public func forward(
        trunks: MLXArray,
        goals: MLXArray,
        state: ManasMLXCoreState? = nil
    ) -> ManasMLXCoreOutput {
        let sequence = normalizeSequence(trunks)
        var embedded = tanh(encoder1(sequence))
        embedded = tanh(encoder2(embedded))

        let goalEmbedded = tanh(goalEncoder(goals))
        let gate = sigmoid(goalGate(goalEmbedded))
        let gatedGoal = goalEmbedded * gate

        let goalBroadcast: MLXArray
        if gatedGoal.ndim == 2 {
            let expanded = expandedDimensions(gatedGoal, axis: 1)
            goalBroadcast = repeated(expanded, count: embedded.dim(1), axis: 1)
        } else if gatedGoal.dim(1) == 1 {
            goalBroadcast = repeated(gatedGoal, count: embedded.dim(1), axis: 1)
        } else {
            goalBroadcast = gatedGoal
        }

        let combined = concatenated([embedded, goalBroadcast], axis: -1)

        let fastSeq = fastGRU(combined, hidden: state?.fast)
        let slowSeq = slowGRU(combined, hidden: state?.slow)
        let merged = concatenated([fastSeq, slowSeq], axis: -1)

        var drives = driveHead(merged)
        drives = tanh(drives) * config.driveScale

        let auxOutput: MLXArray?
        if config.auxEnabled {
            auxOutput = sanitizeAux(auxHead(merged))
        } else {
            auxOutput = nil
        }

        let lastIndex = merged.dim(-2) - 1
        let nextFast = fastSeq[.ellipsis, lastIndex, 0...]
        let nextSlow = slowSeq[.ellipsis, lastIndex, 0...]

        return ManasMLXCoreOutput(
            drives: drives,
            nextState: ManasMLXCoreState(fast: nextFast, slow: nextSlow),
            aux: auxOutput
        )
    }

    public func callAsFunction(_ trunks: MLXArray, goals: MLXArray) -> MLXArray {
        forward(trunks: trunks, goals: goals).drives
    }

    private func normalizeSequence(_ input: MLXArray) -> MLXArray {
        MLXModelUtils.normalizeSequence(input)
    }

    private func sanitizeAux(_ aux: MLXArray) -> MLXArray? {
        MLXModelUtils.sanitizeAux(aux, expectedSize: config.auxSize)
    }
}
