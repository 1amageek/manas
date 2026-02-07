import MLX
import MLXNN

public struct HybridReflexOutput {
    public let analyticalPD: MLXArray
    public let nnResidual: MLXArray
    public let combined: MLXArray

    public init(analyticalPD: MLXArray, nnResidual: MLXArray, combined: MLXArray) {
        self.analyticalPD = analyticalPD
        self.nnResidual = nnResidual
        self.combined = combined
    }
}

public final class ManasMLXHybridReflex: Module {
    public let config: ManasMLXHybridReflexConfig

    @ModuleInfo public var residualLinear1: Linear
    @ModuleInfo public var residualLinear2: Linear

    public init(config: ManasMLXHybridReflexConfig) {
        self.config = config
        self._residualLinear1.wrappedValue = Linear(config.inputSize, config.residualHiddenSize)
        self._residualLinear2.wrappedValue = Linear(config.residualHiddenSize, config.actuatorCount)
    }

    public func forward(features: MLXArray, omegaError: MLXArray) -> HybridReflexOutput {
        let analyticalPD = omegaError * config.kdGain
        let rawResidual = residualLinear2(tanh(residualLinear1(features)))
        let nnResidual = clip(rawResidual, min: -config.residualClipRange, max: config.residualClipRange)
        let combined = analyticalPD + nnResidual
        return HybridReflexOutput(analyticalPD: analyticalPD, nnResidual: nnResidual, combined: combined)
    }

    public func callAsFunction(_ features: MLXArray, omegaError: MLXArray) -> MLXArray {
        forward(features: features, omegaError: omegaError).combined
    }
}
