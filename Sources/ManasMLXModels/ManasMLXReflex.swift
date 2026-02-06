import MLX
import MLXNN

public final class ManasMLXReflex: Module {
    public let config: ManasMLXReflexConfig

    @ModuleInfo public var clampHead: Linear
    @ModuleInfo public var dampingHead: Linear
    @ModuleInfo public var deltaHead: Linear

    public init(config: ManasMLXReflexConfig) {
        self.config = config
        self._clampHead.wrappedValue = Linear(config.inputSize, config.driveCount)
        self._dampingHead.wrappedValue = Linear(config.inputSize, config.driveCount)
        self._deltaHead.wrappedValue = Linear(config.inputSize, config.driveCount)
    }

    public func forward(_ input: MLXArray) -> ManasMLXReflexOutput {
        let clamp = sigmoid(clampHead(input))
        let damping = tanh(dampingHead(input)) * config.dampingScale
        let delta = tanh(deltaHead(input)) * config.deltaScale
        return ManasMLXReflexOutput(clamp: clamp, damping: damping, delta: delta)
    }

    public func callAsFunction(_ input: MLXArray) -> ManasMLXReflexOutput {
        forward(input)
    }
}
