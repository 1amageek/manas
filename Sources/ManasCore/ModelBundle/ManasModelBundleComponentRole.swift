public enum ManasModelBundleComponentRole: String, Codable, Sendable, Equatable, CaseIterable {
    case modelConfig
    case embodimentContract
    case coreWeights
    case reflexWeights
    case worldModelWeights
    case normalizationStats
    case safetyEnvelope
    case motorBounds
    case trainingManifest
    case trainingMetrics
    case lineage
    case validationArtifact

    public var allowsMultipleComponents: Bool {
        switch self {
        case .trainingMetrics, .validationArtifact:
            return true
        case .modelConfig, .embodimentContract, .coreWeights, .reflexWeights, .worldModelWeights,
             .normalizationStats, .safetyEnvelope, .motorBounds, .trainingManifest, .lineage:
            return false
        }
    }
}
