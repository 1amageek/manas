import MLX
import MLXNN

public final class ManasMLXTemporalCore: Module {
    public let config: ManasMLXTemporalCoreConfig

    @ModuleInfo public var encoder1: Linear
    @ModuleInfo public var encoder2: Linear
    @ModuleInfo public var fastGRU: GRU
    @ModuleInfo public var slowGRU: GRU
    @ModuleInfo public var horizonHead: Linear
    @ModuleInfo public var auxHead: Linear

    public init(config: ManasMLXTemporalCoreConfig) {
        self.config = config
        self._encoder1.wrappedValue = Linear(config.inputSize, config.embeddingSize)
        self._encoder2.wrappedValue = Linear(config.embeddingSize, config.embeddingSize)
        self._fastGRU.wrappedValue = GRU(inputSize: config.embeddingSize, hiddenSize: config.fastHiddenSize)
        self._slowGRU.wrappedValue = GRU(inputSize: config.embeddingSize, hiddenSize: config.slowHiddenSize)
        let combinedHidden = config.fastHiddenSize + config.slowHiddenSize
        self._horizonHead.wrappedValue = Linear(combinedHidden, config.maxHorizonSteps * config.driveCount)
        self._auxHead.wrappedValue = Linear(combinedHidden, max(config.auxSize, 1))
    }

    public func forward(
        trunks: MLXArray,
        state: ManasMLXCoreState? = nil,
        horizonSteps: Int? = nil
    ) -> ManasMLXTemporalCoreOutput {
        let steps = horizonSteps ?? config.maxHorizonSteps
        let sequence = normalizeSequence(trunks)
        var embedded = tanh(encoder1(sequence))
        embedded = tanh(encoder2(embedded))

        let fastSeq = fastGRU(embedded, hidden: state?.fast)
        let slowSeq = slowGRU(embedded, hidden: state?.slow)
        let combined = concatenated([fastSeq, slowSeq], axis: -1)

        let lastIndex = combined.dim(-2) - 1
        let lastHidden = combined[.ellipsis, lastIndex, 0...]

        let rawHorizon = horizonHead(lastHidden)
        let fullHorizon = rawHorizon.reshaped([-1, config.maxHorizonSteps, config.driveCount])
        let horizon = tanh(fullHorizon[.ellipsis, 0..<steps, 0...]) * config.driveScale

        let auxOutput: MLXArray?
        if config.auxEnabled {
            auxOutput = sanitizeAux(auxHead(lastHidden))
        } else {
            auxOutput = nil
        }

        let nextFast = fastSeq[.ellipsis, lastIndex, 0...]
        let nextSlow = slowSeq[.ellipsis, lastIndex, 0...]

        return ManasMLXTemporalCoreOutput(
            horizon: horizon,
            nextState: ManasMLXCoreState(fast: nextFast, slow: nextSlow),
            aux: auxOutput
        )
    }

    public func callAsFunction(_ trunks: MLXArray) -> MLXArray {
        forward(trunks: trunks).horizon
    }

    private func normalizeSequence(_ input: MLXArray) -> MLXArray {
        MLXModelUtils.normalizeSequence(input)
    }

    private func sanitizeAux(_ aux: MLXArray) -> MLXArray? {
        MLXModelUtils.sanitizeAux(aux, expectedSize: config.auxSize)
    }
}

public struct ManasMLXTemporalCoreOutput {
    public let horizon: MLXArray
    public let nextState: ManasMLXCoreState
    public let aux: MLXArray?

    public init(horizon: MLXArray, nextState: ManasMLXCoreState, aux: MLXArray?) {
        self.horizon = horizon
        self.nextState = nextState
        self.aux = aux
    }
}
