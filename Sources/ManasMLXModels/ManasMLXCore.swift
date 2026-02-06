import MLX
import MLXNN

public final class ManasMLXCore: Module {
    public let config: ManasMLXCoreConfig

    @ModuleInfo public var encoder1: Linear
    @ModuleInfo public var encoder2: Linear
    @ModuleInfo public var fastGRU: GRU
    @ModuleInfo public var slowGRU: GRU
    @ModuleInfo public var driveHead: Linear
    @ModuleInfo public var auxHead: Linear

    public init(config: ManasMLXCoreConfig) {
        self.config = config
        self._encoder1.wrappedValue = Linear(config.inputSize, config.embeddingSize)
        self._encoder2.wrappedValue = Linear(config.embeddingSize, config.embeddingSize)
        self._fastGRU.wrappedValue = GRU(inputSize: config.embeddingSize, hiddenSize: config.fastHiddenSize)
        self._slowGRU.wrappedValue = GRU(inputSize: config.embeddingSize, hiddenSize: config.slowHiddenSize)
        self._driveHead.wrappedValue = Linear(config.fastHiddenSize + config.slowHiddenSize, config.driveCount)
        self._auxHead.wrappedValue = Linear(config.fastHiddenSize + config.slowHiddenSize, config.auxSize)
    }

    public func forward(trunks: MLXArray, state: ManasMLXCoreState? = nil) -> ManasMLXCoreOutput {
        let sequence = normalizeSequence(trunks)
        var embedded = encoder1(sequence)
        embedded = tanh(embedded)
        embedded = encoder2(embedded)
        embedded = tanh(embedded)

        let fastSeq = fastGRU(embedded, hidden: state?.fast)
        let slowSeq = slowGRU(embedded, hidden: state?.slow)
        let combined = concatenated([fastSeq, slowSeq], axis: -1)

        var drives = driveHead(combined)
        drives = tanh(drives) * config.driveScale

        let auxOutput: MLXArray?
        if config.auxEnabled {
            auxOutput = sanitizeAux(auxHead(combined))
        } else {
            auxOutput = nil
        }

        let lastIndex = combined.dim(-2) - 1
        let nextFast = fastSeq[.ellipsis, lastIndex, 0...]
        let nextSlow = slowSeq[.ellipsis, lastIndex, 0...]

        return ManasMLXCoreOutput(
            drives: drives,
            nextState: ManasMLXCoreState(fast: nextFast, slow: nextSlow),
            aux: auxOutput
        )
    }

    public func callAsFunction(_ trunks: MLXArray) -> MLXArray {
        forward(trunks: trunks, state: nil).drives
    }

    private func normalizeSequence(_ input: MLXArray) -> MLXArray {
        switch input.ndim {
        case 1:
            return input.reshaped([1, 1, input.shape[0]])
        case 2:
            return input.reshaped([1, input.shape[0], input.shape[1]])
        default:
            return input
        }
    }

    private func sanitizeAux(_ aux: MLXArray) -> MLXArray? {
        guard aux.shape.last == config.auxSize else { return nil }
        let invalid = logicalOr(isNaN(aux), isInf(aux))
        if any(invalid).item(Bool.self) {
            return nil
        }
        return aux
    }
}
