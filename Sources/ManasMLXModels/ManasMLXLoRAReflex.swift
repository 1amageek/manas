import MLX
import MLXNN

public final class ManasMLXLoRAReflex: Module {
    public let reflexConfig: ManasMLXReflexConfig
    public let loraConfig: LoRAConfig

    @ModuleInfo public var clampHead: LoRALinear
    @ModuleInfo public var dampingHead: LoRALinear
    @ModuleInfo public var deltaHead: LoRALinear

    public init(from reflex: ManasMLXReflex, loraConfig: LoRAConfig) {
        self.reflexConfig = reflex.config
        self.loraConfig = loraConfig
        self._clampHead.wrappedValue = LoRALinear(
            base: reflex.clampHead, rank: loraConfig.rank, alpha: loraConfig.alpha
        )
        self._dampingHead.wrappedValue = LoRALinear(
            base: reflex.dampingHead, rank: loraConfig.rank, alpha: loraConfig.alpha
        )
        self._deltaHead.wrappedValue = LoRALinear(
            base: reflex.deltaHead, rank: loraConfig.rank, alpha: loraConfig.alpha
        )
    }

    public func forward(_ input: MLXArray) -> ManasMLXReflexOutput {
        let clamp = sigmoid(clampHead(input))
        let damping = tanh(dampingHead(input)) * reflexConfig.dampingScale
        let delta = tanh(deltaHead(input)) * reflexConfig.deltaScale
        return ManasMLXReflexOutput(clamp: clamp, damping: damping, delta: delta)
    }

    public func callAsFunction(_ input: MLXArray) -> ManasMLXReflexOutput {
        forward(input)
    }
}
