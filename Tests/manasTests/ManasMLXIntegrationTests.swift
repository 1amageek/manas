import Testing
@testable import ManasMLXModels

@Test func sharedEncoderMorphologyOnlyConfigKeepsGruInputAligned() {
    let config = ManasMLXCoreConfig(
        inputSize: 4,
        embeddingSize: 32,
        fastHiddenSize: 16,
        slowHiddenSize: 8,
        driveCount: 2,
        auxSize: 0,
        typeEmbeddingDim: 4,
        typeEmbeddingCount: 16,
        ascendingTypeIndices: [0, 1, 2, 3],
        useSharedEncoder: true,
        useSharedDecoder: false,
        morphologyDim: 3
    )

    #expect(config.gruInputSize == 64)
}

@Test func loraAdapterClassifiesSharedCoreAsNerve() {
    let config = ManasMLXCoreConfig(
        inputSize: 4,
        embeddingSize: 32,
        fastHiddenSize: 16,
        slowHiddenSize: 8,
        driveCount: 2,
        auxSize: 0,
        typeEmbeddingDim: 4,
        typeEmbeddingCount: 16,
        ascendingTypeIndices: [0, 1, 2, 3],
        actuatorTypeIndices: [4, 5],
        useSharedEncoder: true,
        useSharedDecoder: true,
        actuatorCount: 2
    )
    #expect(LoRAAdapter.coreAdapterKind(for: config) == .nerve)
}

@Test func loraAdapterClassifiesLegacyCoreAsLegacy() {
    let config = ManasMLXCoreConfig(
        inputSize: 4,
        embeddingSize: 32,
        fastHiddenSize: 16,
        slowHiddenSize: 8,
        driveCount: 2,
        auxSize: 0
    )
    #expect(LoRAAdapter.coreAdapterKind(for: config) == .legacy)
}
