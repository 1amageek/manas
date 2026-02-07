import MLX
import MLXNN

public final class SharedEncoder: Module {
    @ModuleInfo public var linear1: Linear
    @ModuleInfo public var linear2: Linear

    public init(inputSize: Int, embeddingSize: Int) {
        self._linear1.wrappedValue = Linear(inputSize, embeddingSize)
        self._linear2.wrappedValue = Linear(embeddingSize, embeddingSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        tanh(linear2(tanh(linear1(x))))
    }
}
