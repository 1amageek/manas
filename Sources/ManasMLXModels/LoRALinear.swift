import MLX
import MLXNN
import MLXRandom

public final class LoRALinear: Module {
    @ModuleInfo public var base: Linear
    @ModuleInfo public var loraA: Linear
    @ModuleInfo public var loraB: Linear

    public let rank: Int
    public let alpha: Float

    public init(inputSize: Int, outputSize: Int, rank: Int, alpha: Float = 16.0) {
        self.rank = rank
        self.alpha = alpha
        self._base.wrappedValue = Linear(inputSize, outputSize)
        self._loraA.wrappedValue = Linear(inputSize, rank, bias: false)
        let zeroWeight = MLXArray.zeros([outputSize, rank])
        self._loraB.wrappedValue = Linear(weight: zeroWeight, bias: nil)
        super.init()
        base.freeze()
    }

    public init(base existingBase: Linear, rank: Int, alpha: Float = 16.0) {
        self.rank = rank
        self.alpha = alpha
        let inputSize = existingBase.weight.dim(1)
        let outputSize = existingBase.weight.dim(0)
        self._base.wrappedValue = existingBase
        self._loraA.wrappedValue = Linear(inputSize, rank, bias: false)
        let zeroWeight = MLXArray.zeros([outputSize, rank])
        self._loraB.wrappedValue = Linear(weight: zeroWeight, bias: nil)
        super.init()
        existingBase.freeze()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let baseOutput = base(x)
        let loraOutput = loraB(loraA(x))
        return baseOutput + loraOutput * (alpha / Float(rank))
    }

    public func merged() -> Linear {
        let scale = alpha / Float(rank)
        let loraWeight = matmul(loraB.weight, loraA.weight) * scale
        let mergedWeight = base.weight + loraWeight
        return Linear(weight: mergedWeight, bias: base.bias)
    }
}
