import ManasMLXTraining
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
