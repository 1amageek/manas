import MLX

public struct LoRAConfig: Sendable, Equatable, Codable {
    public let rank: Int
    public let alpha: Float
    public let targetModules: [String]

    public init(
        rank: Int = 8,
        alpha: Float = 16.0,
        targetModules: [String] = ["encoder1", "encoder2", "driveHead"]
    ) {
        self.rank = rank
        self.alpha = alpha
        self.targetModules = targetModules
    }
}
