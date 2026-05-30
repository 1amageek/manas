import Foundation

public enum ManasModelBundleValidationError: Error, Sendable, Equatable, CustomStringConvertible {
    case missingManifest(URL)
    case unsupportedSchemaVersion(Int)
    case emptyBundleID
    case emptyModelFamily
    case emptyEmbodimentHash
    case emptyConfigHash
    case emptyObservationSchemaID
    case emptyDriveSemanticsID
    case invalidCorePeriod(Double)
    case invalidReflexPeriod(Double)
    case reflexNotFasterThanCore(core: Double, reflex: Double)
    case missingRequiredRole(ManasModelBundleComponentRole)
    case duplicateSingletonRole(ManasModelBundleComponentRole)
    case invalidComponentPath(String)
    case emptyContentType(String)
    case missingRequiredComponent(String)
    case byteCountMismatch(path: String, expected: Int, actual: Int)
    case digestMismatch(path: String, expected: String, actual: String)

    public var description: String {
        switch self {
        case .missingManifest(let url):
            return "missing-manifest(\(url.path))"
        case .unsupportedSchemaVersion(let version):
            return "unsupported-schema-version(\(version))"
        case .emptyBundleID:
            return "empty-bundle-id"
        case .emptyModelFamily:
            return "empty-model-family"
        case .emptyEmbodimentHash:
            return "empty-embodiment-hash"
        case .emptyConfigHash:
            return "empty-config-hash"
        case .emptyObservationSchemaID:
            return "empty-observation-schema-id"
        case .emptyDriveSemanticsID:
            return "empty-drive-semantics-id"
        case .invalidCorePeriod(let value):
            return "invalid-core-period(\(value))"
        case .invalidReflexPeriod(let value):
            return "invalid-reflex-period(\(value))"
        case .reflexNotFasterThanCore(let core, let reflex):
            return "reflex-not-faster-than-core(core: \(core), reflex: \(reflex))"
        case .missingRequiredRole(let role):
            return "missing-required-role(\(role.rawValue))"
        case .duplicateSingletonRole(let role):
            return "duplicate-singleton-role(\(role.rawValue))"
        case .invalidComponentPath(let path):
            return "invalid-component-path(\(path))"
        case .emptyContentType(let path):
            return "empty-content-type(\(path))"
        case .missingRequiredComponent(let path):
            return "missing-required-component(\(path))"
        case .byteCountMismatch(let path, let expected, let actual):
            return "byte-count-mismatch(path: \(path), expected: \(expected), actual: \(actual))"
        case .digestMismatch(let path, let expected, let actual):
            return "digest-mismatch(path: \(path), expected: \(expected), actual: \(actual))"
        }
    }
}
