import MLX
import MLXNN

final class AffineDeltaModel: Module, UnaryLayer {
    @ModuleInfo var linear: Linear

    init(featureCount: Int) {
        self.linear = Linear(featureCount, 1, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(x)
    }
}
