import MLX

@available(*, deprecated, message: "Use ManasMLXCoreConfig with descendingSize > 0 instead")
public struct ManasMLXGoalCoreConfig: Sendable, Equatable, Codable {
    public let inputSize: Int
    public let embeddingSize: Int
    public let fastHiddenSize: Int
    public let slowHiddenSize: Int
    public let goalSize: Int
    public let goalEmbeddingSize: Int
    public let driveCount: Int
    public let driveScale: Float
    public let auxSize: Int
    public let auxEnabled: Bool

    public init(
        inputSize: Int,
        embeddingSize: Int,
        fastHiddenSize: Int,
        slowHiddenSize: Int,
        goalSize: Int,
        goalEmbeddingSize: Int? = nil,
        driveCount: Int,
        driveScale: Float = 1.0,
        auxSize: Int = 0,
        auxEnabled: Bool = false
    ) {
        self.inputSize = inputSize
        self.embeddingSize = embeddingSize
        self.fastHiddenSize = fastHiddenSize
        self.slowHiddenSize = slowHiddenSize
        self.goalSize = goalSize
        self.goalEmbeddingSize = goalEmbeddingSize ?? embeddingSize
        self.driveCount = driveCount
        self.driveScale = driveScale
        self.auxSize = auxSize
        self.auxEnabled = auxEnabled
    }
}
