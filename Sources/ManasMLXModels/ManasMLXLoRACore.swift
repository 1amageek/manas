import MLX
import MLXNN

public final class ManasMLXLoRACore: Module {
    public let coreConfig: ManasMLXCoreConfig
    public let loraConfig: LoRAConfig

    @ModuleInfo public var encoder1: LoRALinear
    @ModuleInfo public var encoder2: LoRALinear
    @ModuleInfo public var fastGRU: GRU
    @ModuleInfo public var slowGRU: GRU
    @ModuleInfo public var driveHead: LoRALinear
    @ModuleInfo public var auxHead: Linear

    public init(from core: ManasMLXCore, loraConfig: LoRAConfig) {
        self.coreConfig = core.config
        self.loraConfig = loraConfig
        self._encoder1.wrappedValue = LoRALinear(
            base: core.encoder1, rank: loraConfig.rank, alpha: loraConfig.alpha
        )
        self._encoder2.wrappedValue = LoRALinear(
            base: core.encoder2, rank: loraConfig.rank, alpha: loraConfig.alpha
        )
        self._fastGRU.wrappedValue = core.fastGRU
        self._slowGRU.wrappedValue = core.slowGRU
        self._driveHead.wrappedValue = LoRALinear(
            base: core.driveHead, rank: loraConfig.rank, alpha: loraConfig.alpha
        )
        self._auxHead.wrappedValue = core.auxHead
        super.init()
        freezeNonLoRA()
    }

    public func forward(trunks: MLXArray, state: ManasMLXCoreState? = nil) -> ManasMLXCoreOutput {
        let sequence = normalizeSequence(trunks)
        var embedded = tanh(encoder1(sequence))
        embedded = tanh(encoder2(embedded))

        let fastSeq = fastGRU(embedded, hidden: state?.fast)
        let slowSeq = slowGRU(embedded, hidden: state?.slow)
        let combined = concatenated([fastSeq, slowSeq], axis: -1)

        var drives = driveHead(combined)
        drives = tanh(drives) * coreConfig.driveScale

        let auxOutput: MLXArray?
        if coreConfig.auxEnabled {
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

    private func freezeNonLoRA() {
        fastGRU.freeze()
        slowGRU.freeze()
        auxHead.freeze()
    }

    private func normalizeSequence(_ input: MLXArray) -> MLXArray {
        MLXModelUtils.normalizeSequence(input)
    }

    private func sanitizeAux(_ aux: MLXArray) -> MLXArray? {
        MLXModelUtils.sanitizeAux(aux, expectedSize: coreConfig.auxSize)
    }
}
