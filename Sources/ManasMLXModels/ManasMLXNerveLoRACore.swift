import MLX
import MLXNN

public final class ManasMLXNerveLoRACore: Module {
    public let coreConfig: ManasMLXCoreConfig
    public let loraConfig: LoRAConfig

    // Shared encoder LoRA (level >= 2)
    @ModuleInfo public var sharedEncoderLinear1: LoRALinear?
    @ModuleInfo public var sharedEncoderLinear2: LoRALinear?

    // Shared decoder LoRA (level >= 1)
    @ModuleInfo public var sharedDecoderLinear1: LoRALinear?
    @ModuleInfo public var sharedDecoderLinear2: LoRALinear?

    // Base model (non-LoRA layers)
    @ModuleInfo public var base: ManasMLXCore

    public init(from core: ManasMLXCore, loraConfig: LoRAConfig) {
        self.coreConfig = core.config
        self.loraConfig = loraConfig
        self._base.wrappedValue = core

        let level = loraConfig.level ?? .decoderOnly

        // Level 1+: Decoder LoRA
        if level.rawValue >= LoRALevel.decoderOnly.rawValue, let decoder = core.sharedActuatorDecoder {
            self._sharedDecoderLinear1.wrappedValue = LoRALinear(
                base: decoder.linear1, rank: loraConfig.rank, alpha: loraConfig.alpha
            )
            self._sharedDecoderLinear2.wrappedValue = LoRALinear(
                base: decoder.linear2, rank: loraConfig.rank, alpha: loraConfig.alpha
            )
        }

        // Level 2+: Encoder LoRA
        if level.rawValue >= LoRALevel.decoderAndEncoder.rawValue, let encoder = core.sharedEncoder {
            self._sharedEncoderLinear1.wrappedValue = LoRALinear(
                base: encoder.linear1, rank: loraConfig.rank, alpha: loraConfig.alpha
            )
            self._sharedEncoderLinear2.wrappedValue = LoRALinear(
                base: encoder.linear2, rank: loraConfig.rank, alpha: loraConfig.alpha
            )
        }

        super.init()
        freezeNonLoRA()
    }

    public func forward(
        trunks: MLXArray,
        descending: MLXArray? = nil,
        morphology: MLXArray? = nil,
        state: ManasMLXCoreState? = nil
    ) -> ManasMLXCoreOutput {
        // Build LoRA encoder function override
        let encoderFn: ((MLXArray) -> MLXArray)?
        if let l1 = sharedEncoderLinear1, let l2 = sharedEncoderLinear2 {
            encoderFn = { x in tanh(l2(tanh(l1(x)))) }
        } else {
            encoderFn = nil
        }

        // Build LoRA decoder function override
        let decoderFn: ((MLXArray) -> MLXArray)?
        if let l1 = sharedDecoderLinear1, let l2 = sharedDecoderLinear2 {
            decoderFn = { x in l2(tanh(l1(x))) }
        } else {
            decoderFn = nil
        }

        let config = base.config
        let (gruInput, embedded) = base.encodeInput(
            trunks: trunks, descending: descending, morphology: morphology,
            encoderFn: encoderFn
        )

        let fastSeq = base.fastGRU(gruInput, hidden: state?.fast)
        let slowSeq = base.slowGRU(gruInput, hidden: state?.slow)
        let h_t = concatenated([fastSeq, slowSeq], axis: -1)

        let combined: MLXArray
        let lastZ: MLXArray?
        if config.rssmEnabled, let posteriorNet = base.posteriorNet {
            let posteriorInput = concatenated([h_t, embedded], axis: -1)
            let posteriorLogits = posteriorNet(posteriorInput)
            let z_t = base.sampleCategorical(logits: posteriorLogits)
            combined = concatenated([h_t, z_t], axis: -1)
            let lastIndex = z_t.dim(-2) - 1
            lastZ = z_t[.ellipsis, lastIndex, 0...]
        } else {
            combined = h_t
            lastZ = nil
        }

        let (features, nextHistory, adaptZ) = base.applyAdaptation(
            combined: combined, embedded: embedded, state: state
        )
        let drives = base.decodeDrives(features, decoderFn: decoderFn)

        let auxOutput: MLXArray?
        if config.auxEnabled {
            auxOutput = MLXModelUtils.sanitizeAux(base.auxHead(features), expectedSize: config.auxSize)
        } else {
            auxOutput = nil
        }

        let lastIndex = h_t.dim(-2) - 1
        let nextFast = fastSeq[.ellipsis, lastIndex, 0...]
        let nextSlow = slowSeq[.ellipsis, lastIndex, 0...]

        return ManasMLXCoreOutput(
            drives: drives,
            nextState: ManasMLXCoreState(
                fast: nextFast, slow: nextSlow, z: lastZ,
                ascendingHistory: nextHistory, adaptationZ: adaptZ
            ),
            aux: auxOutput
        )
    }

    public func callAsFunction(_ trunks: MLXArray) -> MLXArray {
        forward(trunks: trunks).drives
    }

    private func freezeNonLoRA() {
        base.freeze()

        // Unfreeze LoRA parameters
        if let l1 = sharedDecoderLinear1 { l1.loraA.unfreeze(); l1.loraB.unfreeze() }
        if let l2 = sharedDecoderLinear2 { l2.loraA.unfreeze(); l2.loraB.unfreeze() }
        if let l1 = sharedEncoderLinear1 { l1.loraA.unfreeze(); l1.loraB.unfreeze() }
        if let l2 = sharedEncoderLinear2 { l2.loraA.unfreeze(); l2.loraB.unfreeze() }
    }
}
