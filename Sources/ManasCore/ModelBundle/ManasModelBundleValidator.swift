import Foundation

public struct ManasModelBundleValidator: Sendable {
    public init() {}

    public func loadAndValidate(
        from bundleRoot: URL,
        manifestFileName: String = ManasModelBundleManifest.defaultFileName,
        policy: ManasModelBundleValidationPolicy = .full
    ) throws -> ManasModelBundleManifest {
        let manifestURL = bundleRoot.appendingPathComponent(manifestFileName, isDirectory: false)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ManasModelBundleValidationError.missingManifest(manifestURL)
        }

        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifest = try decoder.decode(ManasModelBundleManifest.self, from: data)
        try validate(manifest, bundleRoot: bundleRoot, policy: policy)
        return manifest
    }

    public func validate(
        _ manifest: ManasModelBundleManifest,
        bundleRoot: URL,
        policy: ManasModelBundleValidationPolicy = .full
    ) throws {
        guard manifest.schemaVersion == ManasModelBundleManifest.currentSchemaVersion else {
            throw ManasModelBundleValidationError.unsupportedSchemaVersion(manifest.schemaVersion)
        }
        guard !manifest.bundleID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ManasModelBundleValidationError.emptyBundleID
        }
        guard !manifest.modelFamily.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ManasModelBundleValidationError.emptyModelFamily
        }
        try validate(runtimeContract: manifest.runtimeContract)
        try validateRequiredRoles(in: manifest.components)
        try validateComponents(manifest.components, bundleRoot: bundleRoot, policy: policy)
    }

    private func validate(runtimeContract: ManasModelBundleRuntimeContract) throws {
        guard !runtimeContract.embodimentHash.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ManasModelBundleValidationError.emptyEmbodimentHash
        }
        guard !runtimeContract.configHash.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ManasModelBundleValidationError.emptyConfigHash
        }
        guard !runtimeContract.observationSchemaID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ManasModelBundleValidationError.emptyObservationSchemaID
        }
        guard !runtimeContract.driveSemanticsID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ManasModelBundleValidationError.emptyDriveSemanticsID
        }

        if let corePeriod = runtimeContract.corePeriodSeconds {
            guard corePeriod.isFinite && corePeriod > 0 else {
                throw ManasModelBundleValidationError.invalidCorePeriod(corePeriod)
            }
        }
        if let reflexPeriod = runtimeContract.reflexPeriodSeconds {
            guard reflexPeriod.isFinite && reflexPeriod > 0 else {
                throw ManasModelBundleValidationError.invalidReflexPeriod(reflexPeriod)
            }
        }
        if let corePeriod = runtimeContract.corePeriodSeconds,
           let reflexPeriod = runtimeContract.reflexPeriodSeconds,
           reflexPeriod >= corePeriod {
            throw ManasModelBundleValidationError.reflexNotFasterThanCore(core: corePeriod, reflex: reflexPeriod)
        }
    }

    private func validateRequiredRoles(in components: [ManasModelBundleComponent]) throws {
        let roles = Set(components.map(\.role))
        for role in [ManasModelBundleComponentRole.modelConfig, .coreWeights] {
            guard roles.contains(role) else {
                throw ManasModelBundleValidationError.missingRequiredRole(role)
            }
        }

        var singletonRoles = Set<ManasModelBundleComponentRole>()
        for component in components where !component.role.allowsMultipleComponents {
            guard singletonRoles.insert(component.role).inserted else {
                throw ManasModelBundleValidationError.duplicateSingletonRole(component.role)
            }
        }
    }

    private func validateComponents(
        _ components: [ManasModelBundleComponent],
        bundleRoot: URL,
        policy: ManasModelBundleValidationPolicy
    ) throws {
        let fileManager = FileManager.default
        for component in components {
            guard isSafeRelativePath(component.path) else {
                throw ManasModelBundleValidationError.invalidComponentPath(component.path)
            }
            guard !component.contentType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ManasModelBundleValidationError.emptyContentType(component.path)
            }

            let url = bundleRoot.appendingPathComponent(component.path, isDirectory: false)
            if component.required {
                guard fileManager.fileExists(atPath: url.path) else {
                    throw ManasModelBundleValidationError.missingRequiredComponent(component.path)
                }
            }
            guard fileManager.fileExists(atPath: url.path) else {
                continue
            }

            if policy.validateComponentByteCounts, let expectedByteCount = component.byteCount {
                let actualByteCount = try byteCount(of: url, fileManager: fileManager)
                guard expectedByteCount == actualByteCount else {
                    throw ManasModelBundleValidationError.byteCountMismatch(
                        path: component.path,
                        expected: expectedByteCount,
                        actual: actualByteCount
                    )
                }
            }
            guard policy.validateComponentDigests, let expectedDigest = component.fnv1a64 else {
                continue
            }
            let data = try Data(contentsOf: url, options: [.mappedIfSafe])
            let actualDigest = Self.fnv1a64Hex(for: data)
            guard expectedDigest == actualDigest else {
                throw ManasModelBundleValidationError.digestMismatch(
                    path: component.path,
                    expected: expectedDigest,
                    actual: actualDigest
                )
            }
        }
    }

    private func byteCount(of url: URL, fileManager: FileManager) throws -> Int {
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        if let number = attributes[.size] as? NSNumber {
            return number.intValue
        }
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        return data.count
    }

    private func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/") else {
            return false
        }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.isEmpty else {
            return false
        }
        return components.allSatisfy { component in
            !component.isEmpty && component != "." && component != ".."
        }
    }

    public static func fnv1a64Hex(for data: Data) -> String {
        let digest = FNV1a64.hash(data: data)
        return String(format: "%016llx", digest)
    }
}
