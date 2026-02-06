/// Semantic category for a drive primitive's purpose.
public enum DriveCategory: String, Hashable, Sendable, Codable {
    case stabilization
    case locomotion
    case manipulation
    case safety
    case tracking
}
