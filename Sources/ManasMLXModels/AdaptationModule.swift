import MLX
import MLXNN

public final class AdaptationModule: Module {
    public let historyLength: Int
    public let inputDim: Int
    public let outputDim: Int

    // Conv1d expects channels-last: [batch, seq_len, channels]
    @ModuleInfo public var conv1: Conv1d
    @ModuleInfo public var conv2: Conv1d
    @ModuleInfo public var linear: Linear

    public init(historyLength: Int = 100, inputDim: Int, outputDim: Int = 16) {
        self.historyLength = historyLength
        self.inputDim = inputDim
        self.outputDim = outputDim

        self._conv1.wrappedValue = Conv1d(inputChannels: inputDim, outputChannels: 32, kernelSize: 8, stride: 4)
        self._conv2.wrappedValue = Conv1d(inputChannels: 32, outputChannels: 16, kernelSize: 4, stride: 2)

        // Output length: historyLength=100 → floor((100-8)/4+1)=24 → floor((24-4)/2+1)=11
        let afterConv1 = (historyLength - 8) / 4 + 1
        let afterConv2 = (afterConv1 - 4) / 2 + 1
        let flatSize = afterConv2 * 16

        self._linear.wrappedValue = Linear(flatSize, outputDim)
    }

    public func callAsFunction(_ history: MLXArray) -> MLXArray {
        // history: [batch, historyLength, inputDim] (channels-last for Conv1d)
        var h = relu(conv1(history))
        h = relu(conv2(h))
        h = h.reshaped([h.dim(0), -1])
        return linear(h)
    }
}
