/// Results from imagination-based actor-critic training.
public struct ImaginationTrainingResult: Sendable {
    /// Actor (driveHead) loss per epoch.
    public let actorLosses: [Float]

    /// Critic (valueHead) loss per epoch.
    public let criticLosses: [Float]

    /// Mean predicted reward across all imagination rollouts.
    public let meanReward: Float

    /// Mean effective horizon length (before continue probability drops).
    public let meanHorizonLength: Float

    public init(
        actorLosses: [Float],
        criticLosses: [Float],
        meanReward: Float,
        meanHorizonLength: Float
    ) {
        self.actorLosses = actorLosses
        self.criticLosses = criticLosses
        self.meanReward = meanReward
        self.meanHorizonLength = meanHorizonLength
    }
}
