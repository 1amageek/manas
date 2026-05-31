import ManasCore
import ManasMLXTraining
import ManasTrainingData
import Testing

@Test func cappedCoreBatchesSampleAcrossDatasetWindowRange() throws {
    let sampler = ManasTrainingWindowSampler(maxBatches: 3)
    let starts = sampler.selectedWindowStarts(maxStart: 8)

    #expect(starts == [0, 4, 8])
}

@Test func singleCappedCoreBatchSamplesMiddleWindow() throws {
    let sampler = ManasTrainingWindowSampler(maxBatches: 1)
    let starts = sampler.selectedWindowStarts(maxStart: 8)

    #expect(starts == [4])
}

@Test func miniBatchingReducesOptimizerStepsByMoreThanHalf() throws {
    let sampler = ManasTrainingWindowSampler()
    let starts = sampler.selectedWindowStarts(maxStart: 17)
    let batches = sampler.selectedWindowStartBatches(maxStart: 17, miniBatchSize: 8)

    #expect(starts.count == 18)
    #expect(batches.count == 3)
    #expect(batches.flatMap { $0 } == starts)
    #expect(batches.count * 2 < starts.count)
}

@Test func cappedMiniBatchingPreservesSampledWindows() throws {
    let sampler = ManasTrainingWindowSampler(maxBatches: 9)
    let starts = sampler.selectedWindowStarts(maxStart: 30)
    let batches = sampler.selectedWindowStartBatches(maxStart: 30, miniBatchSize: 4)

    #expect(starts.count == 9)
    #expect(batches.count == 3)
    #expect(batches.flatMap { $0 } == starts)
    #expect(batches.count * 2 < starts.count)
}

@Test func zeroWindowLimitProducesNoSamples() throws {
    let sampler = ManasTrainingWindowSampler(maxBatches: 0)

    #expect(sampler.selectedWindowStarts(maxStart: 10).isEmpty)
    #expect(sampler.selectedWindowStartBatches(maxStart: 10, miniBatchSize: 4).isEmpty)
}

@Test func coreBatchBuilderHonorsMiniBatchSizeInTensorShape() throws {
    let dataset = makeDataset(recordCount: 6, channelCount: 2, driveCount: 1)
    var pipeline = ManasTrunkPipeline(
        bundle: PassThroughNerveBundle(configuration: .init(channelCount: 2)),
        gate: IdentityGating(),
        trunks: BasicTrunksBuilder()
    )
    let builder = ManasTrainingBatchBuilder(
        sequenceLength: 2,
        driveCount: 1,
        miniBatchSize: 2
    )

    let batches = try builder.makeCoreBatches(dataset: dataset, pipeline: &pipeline)

    #expect(batches.count == 3)
    #expect(batches[0].trunks.shape == [2, 2, 8])
    #expect(batches[0].targetDrives.shape == [2, 2, 1])
    #expect(batches[2].trunks.shape == [1, 2, 8])
}

@Test func normalizedMiniBatchSizeUsesDefaultForMissingOrInvalidValues() {
    #expect(ManasTrainingBatchBuilder.normalizedMiniBatchSize(nil) == ManasTrainingBatchBuilder.defaultMiniBatchSize)
    #expect(ManasTrainingBatchBuilder.normalizedMiniBatchSize(0) == 1)
    #expect(ManasTrainingBatchBuilder.normalizedMiniBatchSize(64) == 64)
}

private func makeDataset(
    recordCount: Int,
    channelCount: Int,
    driveCount: Int
) -> ManasTrainingDataset {
    var records: [ManasTrainingDatasetRecord] = []
    records.reserveCapacity(recordCount)
    for index in 0..<recordCount {
        var sensors: [ManasTrainingSensorSample] = []
        sensors.reserveCapacity(channelCount)
        for channel in 0..<channelCount {
            sensors.append(ManasTrainingSensorSample(
                channelIndex: UInt32(channel),
                value: Double(index + channel),
                timestamp: Double(index) * 0.01
            ))
        }

        var driveIntents: [ManasTrainingDriveIntent] = []
        driveIntents.reserveCapacity(driveCount)
        for drive in 0..<driveCount {
            driveIntents.append(ManasTrainingDriveIntent(
                driveIndex: UInt32(drive),
                value: Double(index + drive) / Double(max(recordCount, 1))
            ))
        }

        records.append(ManasTrainingDatasetRecord(
            time: Double(index) * 0.01,
            sensors: sensors,
            driveIntents: driveIntents,
            reflexCorrections: []
        ))
    }
    let metadata = ManasTrainingDatasetMetadata(
        scenarioId: "mini-batch-shape",
        seed: 1,
        timeStep: 0.01,
        determinismTier: "test",
        configHash: "test",
        channelCount: channelCount,
        driveCount: driveCount,
        recordCount: recordCount
    )
    return ManasTrainingDataset(metadata: metadata, records: records)
}
