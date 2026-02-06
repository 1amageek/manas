/// Semantic metadata attached to a drive primitive, describing its purpose and priority.
public struct DriveSemantics: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case nonFinitePriority
    }

    public let label: String
    public let category: DriveCategory
    public let priority: Double

    public init(label: String, category: DriveCategory, priority: Double = 0.5) throws {
        guard priority.isFinite else {
            throw ValidationError.nonFinitePriority
        }
        self.label = label
        self.category = category
        self.priority = min(max(priority, 0.0), 1.0)
    }
}
