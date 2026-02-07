import MLX
import MLXNN

public final class SharedActuatorDecoder: Module {
    @ModuleInfo public var linear1: Linear
    @ModuleInfo public var linear2: Linear

    public init(inputSize: Int, hiddenSize: Int = 128) {
        self._linear1.wrappedValue = Linear(inputSize, hiddenSize)
        self._linear2.wrappedValue = Linear(hiddenSize, 1)
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(tanh(linear1(input)))
    }
}
