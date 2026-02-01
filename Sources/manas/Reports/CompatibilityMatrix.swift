public struct CompatibilityMatrix: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case empty(String)
    }

    public let manasVersion: String
    public let kuyukaiVersion: String
    public let conformanceSuiteVersion: String
    public let scenarioSuiteId: String
    public let determinismTier: String
    public let tolerances: String
    public let badges: [Badge]
    public let oedId: String
    public let oedVersion: String

    public init(
        manasVersion: String,
        kuyukaiVersion: String,
        conformanceSuiteVersion: String,
        scenarioSuiteId: String,
        determinismTier: String,
        tolerances: String,
        badges: [Badge],
        oedId: String,
        oedVersion: String
    ) throws {
        guard !manasVersion.isEmpty else { throw ValidationError.empty("manasVersion") }
        guard !kuyukaiVersion.isEmpty else { throw ValidationError.empty("kuyukaiVersion") }
        guard !conformanceSuiteVersion.isEmpty else { throw ValidationError.empty("conformanceSuiteVersion") }
        guard !scenarioSuiteId.isEmpty else { throw ValidationError.empty("scenarioSuiteId") }
        guard !determinismTier.isEmpty else { throw ValidationError.empty("determinismTier") }
        guard !tolerances.isEmpty else { throw ValidationError.empty("tolerances") }
        guard !badges.isEmpty else { throw ValidationError.empty("badges") }
        guard !oedId.isEmpty else { throw ValidationError.empty("oedId") }
        guard !oedVersion.isEmpty else { throw ValidationError.empty("oedVersion") }

        self.manasVersion = manasVersion
        self.kuyukaiVersion = kuyukaiVersion
        self.conformanceSuiteVersion = conformanceSuiteVersion
        self.scenarioSuiteId = scenarioSuiteId
        self.determinismTier = determinismTier
        self.tolerances = tolerances
        self.badges = badges
        self.oedId = oedId
        self.oedVersion = oedVersion
    }
}

