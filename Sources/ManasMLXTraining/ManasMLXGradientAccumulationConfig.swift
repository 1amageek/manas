import MLX

public struct ManasMLXGradientAccumulationConfig: Sendable, Equatable {
    public let microBatchSize: Int
    public let accumulationSteps: Int
    public let learningRate: Float
    public let maxGradNorm: Float?
    public let driveLossWeight: Float
    public let auxLossWeight: Float

    public var effectiveBatchSize: Int {
        microBatchSize * accumulationSteps
    }

    public init(
        microBatchSize: Int = 4,
        accumulationSteps: Int = 8,
        learningRate: Float = 0.001,
        maxGradNorm: Float? = 1.0,
        driveLossWeight: Float = 1.0,
        auxLossWeight: Float = 0.1
    ) {
        self.microBatchSize = microBatchSize
        self.accumulationSteps = accumulationSteps
        self.learningRate = learningRate
        self.maxGradNorm = maxGradNorm
        self.driveLossWeight = driveLossWeight
        self.auxLossWeight = auxLossWeight
    }
}
