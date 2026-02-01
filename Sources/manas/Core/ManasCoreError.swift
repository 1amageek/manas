public enum ManasCoreError: Error, Equatable {
    case missingWeight(PerceptionIndex)
    case missingThreshold(PerceptionIndex)
    case invalidInhibitionFactor(Double)
    case invalidGlobalInhibition(Double)
    case reflexNotHandled([PerceptionIndex])
}
