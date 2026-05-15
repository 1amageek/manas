import Foundation

public struct ManasModelBundleManifest: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1
    public static let defaultFileName = "manas-bundle.json"

    public let schemaVersion: Int
    public let bundleID: String
    public let modelFamily: String
    public let createdAt: Date
    public let parentBundleID: String?
    public let runtimeContract: ManasModelBundleRuntimeContract
    public let components: [ManasModelBundleComponent]

    public init(
        schemaVersion: Int = ManasModelBundleManifest.currentSchemaVersion,
        bundleID: String,
        modelFamily: String = "manas",
        createdAt: Date,
        parentBundleID: String? = nil,
        runtimeContract: ManasModelBundleRuntimeContract,
        components: [ManasModelBundleComponent]
    ) {
        self.schemaVersion = schemaVersion
        self.bundleID = bundleID
        self.modelFamily = modelFamily
        self.createdAt = createdAt
        self.parentBundleID = parentBundleID
        self.runtimeContract = runtimeContract
        self.components = components
    }
}
