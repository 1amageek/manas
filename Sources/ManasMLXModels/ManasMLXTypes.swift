import MLX

public struct ManasMLXCoreConfig: Sendable, Equatable, Codable {
    public let inputSize: Int
    public let embeddingSize: Int
    public let fastHiddenSize: Int
    public let slowHiddenSize: Int
    public let driveCount: Int
    public let driveScale: Float
    public let auxSize: Int
    public let auxEnabled: Bool

    public init(
        inputSize: Int,
        embeddingSize: Int,
        fastHiddenSize: Int,
        slowHiddenSize: Int,
        driveCount: Int,
        driveScale: Float = 1.0,
        auxSize: Int,
        auxEnabled: Bool = false
    ) {
        self.inputSize = inputSize
        self.embeddingSize = embeddingSize
        self.fastHiddenSize = fastHiddenSize
        self.slowHiddenSize = slowHiddenSize
        self.driveCount = driveCount
        self.driveScale = driveScale
        self.auxSize = auxSize
        self.auxEnabled = auxEnabled
    }
}

public struct ManasMLXCoreState {
    public let fast: MLXArray
    public let slow: MLXArray

    public init(fast: MLXArray, slow: MLXArray) {
        self.fast = fast
        self.slow = slow
    }
}

public struct ManasMLXCoreOutput {
    public let drives: MLXArray
    public let nextState: ManasMLXCoreState
    public let aux: MLXArray?

    public init(drives: MLXArray, nextState: ManasMLXCoreState, aux: MLXArray?) {
        self.drives = drives
        self.nextState = nextState
        self.aux = aux
    }
}

public struct ManasMLXReflexConfig: Sendable, Equatable, Codable {
    public let inputSize: Int
    public let driveCount: Int
    public let dampingScale: Float
    public let deltaScale: Float

    public init(
        inputSize: Int,
        driveCount: Int,
        dampingScale: Float = 0.2,
        deltaScale: Float = 0.1
    ) {
        self.inputSize = inputSize
        self.driveCount = driveCount
        self.dampingScale = dampingScale
        self.deltaScale = deltaScale
    }
}

public struct ManasMLXReflexOutput {
    public let clamp: MLXArray
    public let damping: MLXArray
    public let delta: MLXArray

    public init(clamp: MLXArray, damping: MLXArray, delta: MLXArray) {
        self.clamp = clamp
        self.damping = damping
        self.delta = delta
    }
}
