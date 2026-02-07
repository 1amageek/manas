import Foundation

public struct ManasMLXHybridReflexConfig: Sendable, Equatable, Codable {
    public let inputSize: Int
    public let actuatorCount: Int
    public let residualHiddenSize: Int
    public let residualClipRange: Float
    public let kdGain: Float

    public init(
        inputSize: Int,
        actuatorCount: Int,
        residualHiddenSize: Int = 16,
        residualClipRange: Float = 0.1,
        kdGain: Float = 0.5
    ) {
        self.inputSize = inputSize
        self.actuatorCount = actuatorCount
        self.residualHiddenSize = residualHiddenSize
        self.residualClipRange = residualClipRange
        self.kdGain = kdGain
    }
}
