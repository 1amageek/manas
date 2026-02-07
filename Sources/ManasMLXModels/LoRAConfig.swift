import MLX

public enum LoRALevel: Int, Sendable, Equatable, Codable {
    case decoderOnly = 1
    case decoderAndEncoder = 2
    case decoderEncoderGRU = 3
    case full = 4
}

public struct LoRAConfig: Sendable, Equatable, Codable {
    public let rank: Int
    public let alpha: Float
    public let targetModules: [String]
    public let level: LoRALevel?

    public init(
        rank: Int = 8,
        alpha: Float = 16.0,
        targetModules: [String] = ["encoder1", "encoder2", "driveHead"],
        level: LoRALevel? = nil
    ) {
        self.rank = rank
        self.alpha = alpha
        self.targetModules = targetModules
        self.level = level
    }
}
