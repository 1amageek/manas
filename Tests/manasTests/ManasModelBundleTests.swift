import Foundation
import Testing
@testable import ManasCore

@Test(.timeLimit(.minutes(1))) func manasModelBundleValidatesCompleteBundle() throws {
    let root = try makeTemporaryBundleRoot()
    try write(Data("{}".utf8), to: root.appendingPathComponent("model.json"))
    try write(Data("core".utf8), to: root.appendingPathComponent("core.safetensors"))
    try write(Data("reflex".utf8), to: root.appendingPathComponent("reflex.safetensors"))
    try write(Data("{\"accepted\":true}".utf8), to: root.appendingPathComponent("validation/accepted-checkpoint.json"))

    let coreData = try Data(contentsOf: root.appendingPathComponent("core.safetensors"))
    let manifest = makeManifest(components: [
        .init(role: .modelConfig, path: "model.json", contentType: "application/json"),
        .init(
            role: .coreWeights,
            path: "core.safetensors",
            contentType: "application/vnd.safetensors",
            byteCount: coreData.count,
            fnv1a64: ManasModelBundleValidator.fnv1a64Hex(for: coreData)
        ),
        .init(role: .reflexWeights, path: "reflex.safetensors", contentType: "application/vnd.safetensors"),
        .init(role: .validationArtifact, path: "validation/accepted-checkpoint.json", contentType: "application/json")
    ])

    try ManasModelBundleWriter().write(manifest, to: root)

    let loaded = try ManasModelBundleValidator().loadAndValidate(from: root)
    #expect(loaded == manifest)
}

@Test(.timeLimit(.minutes(1))) func manasModelBundleRejectsMissingRequiredCoreWeights() throws {
    let root = try makeTemporaryBundleRoot()
    try write(Data("{}".utf8), to: root.appendingPathComponent("model.json"))
    let manifest = makeManifest(components: [
        .init(role: .modelConfig, path: "model.json", contentType: "application/json"),
        .init(role: .coreWeights, path: "core.safetensors", contentType: "application/vnd.safetensors")
    ])
    try ManasModelBundleWriter().write(manifest, to: root)

    #expect(throws: ManasModelBundleValidationError.missingRequiredComponent("core.safetensors")) {
        try ManasModelBundleValidator().loadAndValidate(from: root)
    }
}

@Test(.timeLimit(.minutes(1))) func manasModelBundleRejectsUnsafeComponentPath() throws {
    let root = try makeTemporaryBundleRoot()
    try write(Data("{}".utf8), to: root.appendingPathComponent("model.json"))
    let manifest = makeManifest(components: [
        .init(role: .modelConfig, path: "model.json", contentType: "application/json"),
        .init(role: .coreWeights, path: "../core.safetensors", contentType: "application/vnd.safetensors")
    ])

    #expect(throws: ManasModelBundleValidationError.invalidComponentPath("../core.safetensors")) {
        try ManasModelBundleValidator().validate(manifest, bundleRoot: root)
    }
}

@Test(.timeLimit(.minutes(1))) func manasModelBundleRejectsDuplicateSingletonRole() throws {
    let root = try makeTemporaryBundleRoot()
    let manifest = makeManifest(components: [
        .init(role: .modelConfig, path: "model.json", contentType: "application/json"),
        .init(role: .coreWeights, path: "core.safetensors", contentType: "application/vnd.safetensors"),
        .init(role: .coreWeights, path: "weights/core-copy.safetensors", contentType: "application/vnd.safetensors")
    ])

    #expect(throws: ManasModelBundleValidationError.duplicateSingletonRole(.coreWeights)) {
        try ManasModelBundleValidator().validate(manifest, bundleRoot: root)
    }
}

@Test(.timeLimit(.minutes(1))) func manasModelBundleRejectsInvalidMultiRateContract() throws {
    let root = try makeTemporaryBundleRoot()
    let manifest = ManasModelBundleManifest(
        bundleID: "bundle",
        createdAt: Date(timeIntervalSince1970: 0),
        runtimeContract: .init(
            embodimentHash: "embodiment",
            configHash: "config",
            observationSchemaID: "lift-8ch",
            driveSemanticsID: "quad-drive",
            corePeriodSeconds: 0.01,
            reflexPeriodSeconds: 0.02
        ),
        components: [
            .init(role: .modelConfig, path: "model.json", contentType: "application/json"),
            .init(role: .coreWeights, path: "core.safetensors", contentType: "application/vnd.safetensors")
        ]
    )

    #expect(throws: ManasModelBundleValidationError.reflexNotFasterThanCore(core: 0.01, reflex: 0.02)) {
        try ManasModelBundleValidator().validate(manifest, bundleRoot: root)
    }
}

@Test(.timeLimit(.minutes(1))) func manasModelBundleRejectsDigestMismatch() throws {
    let root = try makeTemporaryBundleRoot()
    try write(Data("{}".utf8), to: root.appendingPathComponent("model.json"))
    try write(Data("core".utf8), to: root.appendingPathComponent("core.safetensors"))
    let manifest = makeManifest(components: [
        .init(role: .modelConfig, path: "model.json", contentType: "application/json"),
        .init(
            role: .coreWeights,
            path: "core.safetensors",
            contentType: "application/vnd.safetensors",
            fnv1a64: "0000000000000000"
        )
    ])
    try ManasModelBundleWriter().write(manifest, to: root)

    let actual = ManasModelBundleValidator.fnv1a64Hex(for: Data("core".utf8))
    #expect(throws: ManasModelBundleValidationError.digestMismatch(
        path: "core.safetensors",
        expected: "0000000000000000",
        actual: actual
    )) {
        try ManasModelBundleValidator().loadAndValidate(from: root)
    }
}

private func makeManifest(components: [ManasModelBundleComponent]) -> ManasModelBundleManifest {
    ManasModelBundleManifest(
        bundleID: "bundle",
        createdAt: Date(timeIntervalSince1970: 0),
        runtimeContract: .init(
            embodimentHash: "embodiment",
            configHash: "config",
            observationSchemaID: "lift-8ch",
            driveSemanticsID: "quad-drive",
            motorNerveProfileID: "quadref",
            corePeriodSeconds: 0.02,
            reflexPeriodSeconds: 0.005
        ),
        components: components
    )
}

private func makeTemporaryBundleRoot() throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("manas-model-bundle-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func write(_ data: Data, to url: URL) throws {
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try data.write(to: url, options: [.atomic])
}
