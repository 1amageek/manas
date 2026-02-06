import MLX

public struct ManasMLXTransformerCoreConfig: Sendable, Equatable, Codable {
    public let inputSize: Int
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let ffnHiddenSize: Int
    public let maxSequenceLength: Int
    public let driveCount: Int
    public let driveScale: Float
    public let auxSize: Int
    public let auxEnabled: Bool
    public let goalSize: Int?
    public let useCausalMask: Bool

    public init(
        inputSize: Int,
        dModel: Int = 256,
        numHeads: Int = 4,
        numLayers: Int = 4,
        ffnHiddenSize: Int? = nil,
        maxSequenceLength: Int = 128,
        driveCount: Int,
        driveScale: Float = 1.0,
        auxSize: Int = 0,
        auxEnabled: Bool = false,
        goalSize: Int? = nil,
        useCausalMask: Bool = false
    ) {
        self.inputSize = inputSize
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.ffnHiddenSize = ffnHiddenSize ?? (dModel * 4)
        self.maxSequenceLength = maxSequenceLength
        self.driveCount = driveCount
        self.driveScale = driveScale
        self.auxSize = auxSize
        self.auxEnabled = auxEnabled
        self.goalSize = goalSize
        self.useCausalMask = useCausalMask
    }
}
