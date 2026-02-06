import MLX
import MLXNN

public final class ManasFFN: Module {
    @ModuleInfo public var linear1: Linear
    @ModuleInfo public var linear2: Linear

    public init(dModel: Int, hiddenSize: Int) {
        self._linear1.wrappedValue = Linear(dModel, hiddenSize)
        self._linear2.wrappedValue = Linear(hiddenSize, dModel)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = gelu(linear1(x))
        return linear2(h)
    }
}
