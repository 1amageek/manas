public struct ActuatorLimits: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case missingLimit(ActuatorIndex)
    }

    private let limits: [ActuatorIndex: ActuatorLimit]

    public init(limits: [ActuatorIndex: ActuatorLimit]) {
        self.limits = limits
    }

    public func limit(for index: ActuatorIndex) -> ActuatorLimit? {
        limits[index]
    }

    public var indices: [ActuatorIndex] {
        limits.keys.sorted { $0.rawValue < $1.rawValue }
    }
}

