import MLX
import MLXNN

public final class MorphologyEncoder: Module {
    @ModuleInfo public var linear: Linear

    public init(morphologyDim: Int, embeddingSize: Int) {
        self._linear.wrappedValue = Linear(morphologyDim, embeddingSize)
    }

    public func callAsFunction(_ morphology: MLXArray) -> MLXArray {
        tanh(linear(morphology))
    }
}
