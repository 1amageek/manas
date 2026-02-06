import Testing
@testable import ManasCore

@Test func imu6NerveBundleNormalizesAndQualifies() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -4.0...4.0, accelRange: -2.0...2.0))
    let samples = [
        try SignalSample(channelIndex: 0, value: 2.0, timestamp: 0.0),
        try SignalSample(channelIndex: 1, value: -2.0, timestamp: 0.0),
        try SignalSample(channelIndex: 2, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 3, value: 1.0, timestamp: 0.0),
        try SignalSample(channelIndex: 4, value: -1.0, timestamp: 0.0),
        try SignalSample(channelIndex: 5, value: 2.0, timestamp: 0.0),
    ]

    let output = try bundle.process(samples: samples, time: 0.0)
    #expect(output.features.count == 6)
    #expect(output.quality.count == 6)
    for value in output.features {
        #expect(value >= -1.0)
        #expect(value <= 1.0)
    }
    for q in output.quality {
        #expect(q >= 0.2)
        #expect(q <= 1.0)
    }
}

@Test func nerveBundleDefaultsMatchSpec() async throws {
    #expect(NerveBundleDefaults.qualityFloor == 0.2)
    #expect(NerveBundleDefaults.transductionGain == 2.0)
    #expect(NerveBundleDefaults.slowTau == 0.05)
    #expect(NerveBundleDefaults.fastTau == 0.005)
    #expect(NerveBundleDefaults.normalizationTau == 0.2)
    #expect(NerveBundleDefaults.lateralInhibitionStrength == 0.2)
    #expect(NerveBundleDefaults.delayPenaltyPerSecond == 0.2)
    #expect(NerveBundleDefaults.missingPenalty == 0.5)
    #expect(NerveBundleDefaults.deltaPenalty == 0.1)
    #expect(NerveBundleDefaults.normalizationEpsilon == 1.0e-6)
}

@Test func imu6NerveBundleAppliesDelayAndMissingPenalty() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -4.0...4.0, accelRange: -2.0...2.0))
    let initialSamples = [
        try SignalSample(channelIndex: 0, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 1, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 2, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 3, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 4, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 5, value: 0.0, timestamp: 0.0),
    ]
    _ = try bundle.process(samples: initialSamples, time: 0.0)

    let output = try bundle.process(samples: [], time: 1.0)
    let expected = 0.4
    #expect(abs(output.quality[0] - expected) < 1.0e-6)
    #expect(output.quality[0] >= 0.2)
}

@Test func imu6NerveBundleAppliesLateralInhibition() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -1.0...1.0, accelRange: -1.0...1.0))
    let samples = [
        try SignalSample(channelIndex: 0, value: 0.01, timestamp: 0.0),
        try SignalSample(channelIndex: 1, value: 1.0, timestamp: 0.0),
        try SignalSample(channelIndex: 2, value: 1.0, timestamp: 0.0),
        try SignalSample(channelIndex: 3, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 4, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 5, value: 0.0, timestamp: 0.0),
    ]

    let output = try bundle.process(samples: samples, time: 0.0)
    #expect(output.features[0] < 0.0)
}

@Test func imu6NerveBundleQualityReflectsSaturationWithoutDeltaPenalty() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -4.0...4.0, accelRange: -2.0...2.0))
    let first = try SignalSample(channelIndex: 0, value: 6.0, timestamp: 0.0)
    _ = try bundle.process(samples: [first], time: 0.0)

    let second = try SignalSample(channelIndex: 0, value: 6.0, timestamp: 0.01)
    let output = try bundle.process(samples: [second], time: 0.01)
    #expect(abs(output.quality[0] - 0.5) < 1.0e-6)
}

@Test func imu6NerveBundleQualityIncludesDeltaPenalty() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -4.0...4.0, accelRange: -2.0...2.0))
    let first = try SignalSample(channelIndex: 0, value: 0.0, timestamp: 0.0)
    _ = try bundle.process(samples: [first], time: 0.0)

    let second = try SignalSample(channelIndex: 0, value: 4.0, timestamp: 0.01)
    let output = try bundle.process(samples: [second], time: 0.01)
    #expect(abs(output.quality[0] - 0.9) < 1.0e-6)
}

@Test func imu6NerveBundleFastTapsTrackStepChange() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -4.0...4.0, accelRange: -2.0...2.0))
    let zeros = [
        try SignalSample(channelIndex: 0, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 1, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 2, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 3, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 4, value: 0.0, timestamp: 0.0),
        try SignalSample(channelIndex: 5, value: 0.0, timestamp: 0.0),
    ]
    _ = try bundle.process(samples: zeros, time: 0.0)

    let step = [
        try SignalSample(channelIndex: 0, value: 4.0, timestamp: 0.01),
        try SignalSample(channelIndex: 1, value: 0.0, timestamp: 0.01),
        try SignalSample(channelIndex: 2, value: 0.0, timestamp: 0.01),
        try SignalSample(channelIndex: 3, value: 0.0, timestamp: 0.01),
        try SignalSample(channelIndex: 4, value: 0.0, timestamp: 0.01),
        try SignalSample(channelIndex: 5, value: 0.0, timestamp: 0.01),
    ]
    let output = try bundle.process(samples: step, time: 0.01)
    #expect(abs(output.fastTaps[0]) > abs(output.features[0]))
    #expect(output.features[0] > 0.0)
}

@Test func imu6NerveBundleFastTapsSettleForConstantSignal() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -4.0...4.0, accelRange: -2.0...2.0))
    let constant = [
        try SignalSample(channelIndex: 0, value: 2.0, timestamp: 0.0),
        try SignalSample(channelIndex: 1, value: 2.0, timestamp: 0.0),
        try SignalSample(channelIndex: 2, value: 2.0, timestamp: 0.0),
        try SignalSample(channelIndex: 3, value: 1.0, timestamp: 0.0),
        try SignalSample(channelIndex: 4, value: 1.0, timestamp: 0.0),
        try SignalSample(channelIndex: 5, value: 1.0, timestamp: 0.0),
    ]
    _ = try bundle.process(samples: constant, time: 0.0)
    let output = try bundle.process(samples: constant, time: 0.01)
    #expect(abs(output.fastTaps[0]) < 1.0e-9)
    #expect(abs(output.fastTaps[3]) < 1.0e-9)
}

@Test func imu6NerveBundleRejectsOutOfRangeChannel() async throws {
    var bundle = Imu6NerveBundle(configuration: .init(gyroRange: -4.0...4.0, accelRange: -2.0...2.0))
    let sample = try SignalSample(channelIndex: 6, value: 0.0, timestamp: 0.0)
    #expect(throws: Imu6NerveBundle.ValidationError.outOfRangeChannel(6)) {
        _ = try bundle.process(samples: [sample], time: 0.0)
    }
}

@Test func passThroughNerveBundleRejectsDuplicateChannelSamples() async throws {
    var bundle = PassThroughNerveBundle(configuration: .init(channelCount: 2))
    let sampleA = try SignalSample(channelIndex: 0, value: 0.1, timestamp: 0.0)
    let sampleB = try SignalSample(channelIndex: 0, value: 0.2, timestamp: 0.0)
    #expect(throws: PassThroughNerveBundle.ValidationError.duplicateChannel(0)) {
        _ = try bundle.process(samples: [sampleA, sampleB], time: 0.0)
    }
}

@Test func passThroughNerveBundleRejectsFutureTimestamp() async throws {
    var bundle = PassThroughNerveBundle(configuration: .init(channelCount: 1))
    let sample = try SignalSample(channelIndex: 0, value: 0.1, timestamp: 1.0)
    #expect(throws: PassThroughNerveBundle.ValidationError.futureTimestamp(1.0)) {
        _ = try bundle.process(samples: [sample], time: 0.0)
    }
}

@Test func passThroughNerveBundleRejectsOutOfRangeChannel() async throws {
    var bundle = PassThroughNerveBundle(configuration: .init(channelCount: 1))
    let sample = try SignalSample(channelIndex: 2, value: 0.1, timestamp: 0.0)
    #expect(throws: PassThroughNerveBundle.ValidationError.outOfRangeChannel(2)) {
        _ = try bundle.process(samples: [sample], time: 0.0)
    }
}
