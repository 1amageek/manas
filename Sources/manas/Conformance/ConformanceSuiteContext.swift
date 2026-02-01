public struct ConformanceSuiteContext: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case empty(String)
    }

    public let oedId: String
    public let oedVersion: String
    public let suiteVersion: String
    public let configHash: String

    public init(
        oedId: String,
        oedVersion: String,
        suiteVersion: String,
        configHash: String
    ) throws {
        guard !oedId.isEmpty else { throw ValidationError.empty("oedId") }
        guard !oedVersion.isEmpty else { throw ValidationError.empty("oedVersion") }
        guard !suiteVersion.isEmpty else { throw ValidationError.empty("suiteVersion") }
        guard !configHash.isEmpty else { throw ValidationError.empty("configHash") }

        self.oedId = oedId
        self.oedVersion = oedVersion
        self.suiteVersion = suiteVersion
        self.configHash = configHash
    }
}

