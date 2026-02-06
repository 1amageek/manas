import MLX

public struct ManasMLXTemporalCoreConfig: Sendable, Equatable, Codable {
    public let inputSize: Int
    public let embeddingSize: Int
    public let fastHiddenSize: Int
    public let slowHiddenSize: Int
    public let maxHorizonSteps: Int
    public let driveCount: Int
    public let driveScale: Float
    public let auxSize: Int
    public let auxEnabled: Bool

    public init(
        inputSize: Int,
        embeddingSize: Int,
        fastHiddenSize: Int,
        slowHiddenSize: Int,
        maxHorizonSteps: Int = 8,
        driveCount: Int,
        driveScale: Float = 1.0,
        auxSize: Int = 0,
        auxEnabled: Bool = false
    ) {
        self.inputSize = inputSize
        self.embeddingSize = embeddingSize
        self.fastHiddenSize = fastHiddenSize
        self.slowHiddenSize = slowHiddenSize
        self.maxHorizonSteps = maxHorizonSteps
        self.driveCount = driveCount
        self.driveScale = driveScale
        self.auxSize = auxSize
        self.auxEnabled = auxEnabled
    }
}
