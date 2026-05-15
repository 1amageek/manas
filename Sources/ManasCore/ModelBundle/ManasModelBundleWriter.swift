import Foundation

public struct ManasModelBundleWriter: Sendable {
    public init() {}

    public func write(
        _ manifest: ManasModelBundleManifest,
        to bundleRoot: URL,
        manifestFileName: String = ManasModelBundleManifest.defaultFileName
    ) throws {
        try FileManager.default.createDirectory(at: bundleRoot, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(manifest)
        try data.write(
            to: bundleRoot.appendingPathComponent(manifestFileName, isDirectory: false),
            options: [.atomic]
        )
    }
}
