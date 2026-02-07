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

    // RSSM stochastic latent (0 = disabled, legacy mode)
    public let stochasticCategories: Int
    public let stochasticClasses: Int
    public let stochasticUnimixRatio: Float

    // Prediction heads (0 = disabled)
    public let rewardHeadHiddenSize: Int
    public let continueHeadHiddenSize: Int
    public let valueHeadHiddenSize: Int
    public let trunkPredictorEnabled: Bool

    // Descending channels (0 = disabled, legacy mode)
    public let descendingSize: Int
    public let descendingEmbeddingSize: Int

    // Type embeddings (0 = disabled, legacy flat-vector mode)
    public let typeEmbeddingDim: Int
    public let typeEmbeddingCount: Int
    public let ascendingTypeIndices: [Int]?
    public let descendingTypeIndices: [Int]?
    public let actuatorTypeIndices: [Int]?

    // Shared encoder/decoder (false = legacy fixed Linear)
    public let useSharedEncoder: Bool
    public let useSharedDecoder: Bool
    public let actuatorCount: Int
    public let decoderHiddenSize: Int
    public let morphologyDim: Int

    // Adaptation module (0 = disabled)
    public let adaptationHistoryLength: Int
    public let adaptationOutputDim: Int

    public var stochasticLatentSize: Int { stochasticCategories * stochasticClasses }
    public var rssmEnabled: Bool { stochasticLatentSize > 0 }
    public var descendingEnabled: Bool { descendingSize > 0 }
    public var typeEmbeddingEnabled: Bool { typeEmbeddingDim > 0 }
    public var deterministicSize: Int { fastHiddenSize + slowHiddenSize }
    public var modelStateSize: Int { deterministicSize + stochasticLatentSize }
    public var sharedEncoderEnabled: Bool { useSharedEncoder && typeEmbeddingEnabled }
    public var sharedDecoderEnabled: Bool { useSharedDecoder && typeEmbeddingEnabled }
    public var adaptationEnabled: Bool { adaptationHistoryLength > 0 && adaptationOutputDim > 0 }
    public var fullFeatureSize: Int {
        let base = rssmEnabled ? modelStateSize : deterministicSize
        return adaptationEnabled ? base + adaptationOutputDim : base
    }
    public var gruInputSize: Int {
        if sharedEncoderEnabled {
            // ascPool + descPool + morphToken (3 × embeddingSize)
            let pools = morphologyDim > 0 ? 3 : (descendingEnabled ? 2 : 1)
            return embeddingSize * pools
        } else if descendingEnabled {
            return embeddingSize + descendingEmbeddingSize
        } else {
            return embeddingSize
        }
    }

    public init(
        inputSize: Int,
        embeddingSize: Int,
        fastHiddenSize: Int,
        slowHiddenSize: Int,
        driveCount: Int,
        driveScale: Float = 1.0,
        auxSize: Int,
        auxEnabled: Bool = false,
        stochasticCategories: Int = 0,
        stochasticClasses: Int = 0,
        stochasticUnimixRatio: Float = 0.01,
        rewardHeadHiddenSize: Int = 0,
        continueHeadHiddenSize: Int = 0,
        valueHeadHiddenSize: Int = 0,
        trunkPredictorEnabled: Bool = false,
        descendingSize: Int = 0,
        descendingEmbeddingSize: Int = 0,
        typeEmbeddingDim: Int = 0,
        typeEmbeddingCount: Int = 0,
        ascendingTypeIndices: [Int]? = nil,
        descendingTypeIndices: [Int]? = nil,
        actuatorTypeIndices: [Int]? = nil,
        useSharedEncoder: Bool = false,
        useSharedDecoder: Bool = false,
        actuatorCount: Int = 0,
        decoderHiddenSize: Int = 128,
        morphologyDim: Int = 0,
        adaptationHistoryLength: Int = 0,
        adaptationOutputDim: Int = 0
    ) {
        self.inputSize = inputSize
        self.embeddingSize = embeddingSize
        self.fastHiddenSize = fastHiddenSize
        self.slowHiddenSize = slowHiddenSize
        self.driveCount = driveCount
        self.driveScale = driveScale
        self.auxSize = auxSize
        self.auxEnabled = auxEnabled
        self.stochasticCategories = stochasticCategories
        self.stochasticClasses = stochasticClasses
        self.stochasticUnimixRatio = stochasticUnimixRatio
        self.rewardHeadHiddenSize = rewardHeadHiddenSize
        self.continueHeadHiddenSize = continueHeadHiddenSize
        self.valueHeadHiddenSize = valueHeadHiddenSize
        self.trunkPredictorEnabled = trunkPredictorEnabled
        self.descendingSize = descendingSize
        self.descendingEmbeddingSize = descendingEmbeddingSize
        self.typeEmbeddingDim = typeEmbeddingDim
        self.typeEmbeddingCount = typeEmbeddingCount
        self.ascendingTypeIndices = ascendingTypeIndices
        self.descendingTypeIndices = descendingTypeIndices
        self.actuatorTypeIndices = actuatorTypeIndices
        self.useSharedEncoder = useSharedEncoder
        self.useSharedDecoder = useSharedDecoder
        self.actuatorCount = actuatorCount
        self.decoderHiddenSize = decoderHiddenSize
        self.morphologyDim = morphologyDim
        self.adaptationHistoryLength = adaptationHistoryLength
        self.adaptationOutputDim = adaptationOutputDim
    }

    private enum CodingKeys: String, CodingKey {
        case inputSize, embeddingSize, fastHiddenSize, slowHiddenSize
        case driveCount, driveScale, auxSize, auxEnabled
        case stochasticCategories, stochasticClasses, stochasticUnimixRatio
        case rewardHeadHiddenSize, continueHeadHiddenSize, valueHeadHiddenSize
        case trunkPredictorEnabled
        case descendingSize, descendingEmbeddingSize
        case typeEmbeddingDim, typeEmbeddingCount
        case ascendingTypeIndices, descendingTypeIndices, actuatorTypeIndices
        case useSharedEncoder, useSharedDecoder, actuatorCount, decoderHiddenSize, morphologyDim
        case adaptationHistoryLength, adaptationOutputDim
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        inputSize = try container.decode(Int.self, forKey: .inputSize)
        embeddingSize = try container.decode(Int.self, forKey: .embeddingSize)
        fastHiddenSize = try container.decode(Int.self, forKey: .fastHiddenSize)
        slowHiddenSize = try container.decode(Int.self, forKey: .slowHiddenSize)
        driveCount = try container.decode(Int.self, forKey: .driveCount)
        driveScale = try container.decode(Float.self, forKey: .driveScale)
        auxSize = try container.decode(Int.self, forKey: .auxSize)
        auxEnabled = try container.decode(Bool.self, forKey: .auxEnabled)
        stochasticCategories = try container.decodeIfPresent(Int.self, forKey: .stochasticCategories) ?? 0
        stochasticClasses = try container.decodeIfPresent(Int.self, forKey: .stochasticClasses) ?? 0
        stochasticUnimixRatio = try container.decodeIfPresent(Float.self, forKey: .stochasticUnimixRatio) ?? 0.01
        rewardHeadHiddenSize = try container.decodeIfPresent(Int.self, forKey: .rewardHeadHiddenSize) ?? 0
        continueHeadHiddenSize = try container.decodeIfPresent(Int.self, forKey: .continueHeadHiddenSize) ?? 0
        valueHeadHiddenSize = try container.decodeIfPresent(Int.self, forKey: .valueHeadHiddenSize) ?? 0
        trunkPredictorEnabled = try container.decodeIfPresent(Bool.self, forKey: .trunkPredictorEnabled) ?? false
        descendingSize = try container.decodeIfPresent(Int.self, forKey: .descendingSize) ?? 0
        descendingEmbeddingSize = try container.decodeIfPresent(Int.self, forKey: .descendingEmbeddingSize) ?? 0
        typeEmbeddingDim = try container.decodeIfPresent(Int.self, forKey: .typeEmbeddingDim) ?? 0
        typeEmbeddingCount = try container.decodeIfPresent(Int.self, forKey: .typeEmbeddingCount) ?? 0
        ascendingTypeIndices = try container.decodeIfPresent([Int].self, forKey: .ascendingTypeIndices)
        descendingTypeIndices = try container.decodeIfPresent([Int].self, forKey: .descendingTypeIndices)
        actuatorTypeIndices = try container.decodeIfPresent([Int].self, forKey: .actuatorTypeIndices)
        useSharedEncoder = try container.decodeIfPresent(Bool.self, forKey: .useSharedEncoder) ?? false
        useSharedDecoder = try container.decodeIfPresent(Bool.self, forKey: .useSharedDecoder) ?? false
        actuatorCount = try container.decodeIfPresent(Int.self, forKey: .actuatorCount) ?? 0
        decoderHiddenSize = try container.decodeIfPresent(Int.self, forKey: .decoderHiddenSize) ?? 128
        morphologyDim = try container.decodeIfPresent(Int.self, forKey: .morphologyDim) ?? 0
        adaptationHistoryLength = try container.decodeIfPresent(Int.self, forKey: .adaptationHistoryLength) ?? 0
        adaptationOutputDim = try container.decodeIfPresent(Int.self, forKey: .adaptationOutputDim) ?? 0
    }
}

public struct ManasMLXCoreState {
    public let fast: MLXArray
    public let slow: MLXArray
    public let z: MLXArray?
    public let ascendingHistory: MLXArray?
    public let adaptationZ: MLXArray?

    public init(fast: MLXArray, slow: MLXArray, z: MLXArray? = nil, ascendingHistory: MLXArray? = nil, adaptationZ: MLXArray? = nil) {
        self.fast = fast
        self.slow = slow
        self.z = z
        self.ascendingHistory = ascendingHistory
        self.adaptationZ = adaptationZ
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
