import MLX
import MLXNN

final class MLPDeltaModel: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(featureCount: Int, hiddenSize: Int) {
        self.fc1 = Linear(featureCount, hiddenSize, bias: true)
        self.fc2 = Linear(hiddenSize, 1, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = relu(fc1(x))
        return fc2(hidden)
    }
}
