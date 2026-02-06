import Foundation

public struct PassThroughNerveBundle: NerveBundle {
    public enum ValidationError: Error, Equatable {
        case nonFinite
        case futureTimestamp(TimeInterval)
        case duplicateChannel(UInt32)
        case outOfRangeChannel(UInt32)
    }
    public struct Configuration: Sendable, Equatable {
        public let channelCount: Int

        public init(channelCount: Int) {
            self.channelCount = max(0, channelCount)
        }
    }

    public var configuration: Configuration
    private var lastProcessTime: TimeInterval?
    private var lastValues: [Double]
    private var lastSampleTimes: [TimeInterval]
    private var slowState: [Double]
    private var fastLowpassState: [Double]
    private var normState: [Double]

    public init(configuration: Configuration) {
        self.configuration = configuration
        let count = configuration.channelCount
        self.lastProcessTime = nil
        self.lastValues = Array(repeating: 0.0, count: count)
        self.lastSampleTimes = Array(repeating: 0.0, count: count)
        self.slowState = Array(repeating: 0.0, count: count)
        self.fastLowpassState = Array(repeating: 0.0, count: count)
        self.normState = Array(repeating: 1.0, count: count)
    }

    public mutating func process(samples: [SignalSample], time: TimeInterval) throws -> NerveBundleOutput {
        guard time.isFinite else { throw ValidationError.nonFinite }
        let count = configuration.channelCount
        if count == 0 {
            if let sample = samples.first {
                throw ValidationError.outOfRangeChannel(sample.channelIndex)
            }
            return NerveBundleOutput(features: [], fastTaps: [], quality: [])
        }

        let dt: Double
        if let lastTime = lastProcessTime, time >= lastTime {
            dt = time - lastTime
        } else {
            dt = 0.0
        }
        lastProcessTime = time

        var values = lastValues
        var present = Array(repeating: false, count: count)
        var timestamps = lastSampleTimes
        let previousValues = lastValues

        var seen: Set<UInt32> = []
        for sample in samples {
            guard sample.value.isFinite else { throw ValidationError.nonFinite }
            guard sample.timestamp <= time else { throw ValidationError.futureTimestamp(sample.timestamp) }
            let idx = Int(sample.channelIndex)
            guard seen.insert(sample.channelIndex).inserted else {
                throw ValidationError.duplicateChannel(sample.channelIndex)
            }
            guard idx >= 0, idx < count else { throw ValidationError.outOfRangeChannel(sample.channelIndex) }
            values[idx] = sample.value
            timestamps[idx] = sample.timestamp
            present[idx] = true
        }

        lastValues = values
        lastSampleTimes = timestamps

        var transduced: [Double] = Array(repeating: 0.0, count: count)
        for idx in 0..<count {
            transduced[idx] = tanh(values[idx] * NerveBundleDefaults.transductionGain)
        }

        let inhibited = applyLateralInhibition(values: transduced)

        let alphaNorm = alpha(for: NerveBundleDefaults.normalizationTau, dt: dt)
        for idx in 0..<count {
            let target = abs(inhibited[idx])
            normState[idx] = normState[idx] + alphaNorm * (target - normState[idx])
        }

        var normalized: [Double] = Array(repeating: 0.0, count: count)
        for idx in 0..<count {
            let denom = max(NerveBundleDefaults.normalizationEpsilon, normState[idx])
            normalized[idx] = clamp(inhibited[idx] / denom, min: -1.0, max: 1.0)
        }

        let alphaSlow = alpha(for: NerveBundleDefaults.slowTau, dt: dt)
        let alphaFast = alpha(for: NerveBundleDefaults.fastTau, dt: dt)
        for idx in 0..<count {
            slowState[idx] = slowState[idx] + alphaSlow * (normalized[idx] - slowState[idx])
            fastLowpassState[idx] = fastLowpassState[idx] + alphaFast * (normalized[idx] - fastLowpassState[idx])
        }

        let fastTaps = zip(normalized, fastLowpassState).map { value, fastLP in
            value - fastLP
        }

        let quality = buildQuality(
            values: values,
            previousValues: previousValues,
            timestamps: timestamps,
            present: present,
            time: time
        )

        return NerveBundleOutput(features: slowState, fastTaps: fastTaps, quality: quality)
    }

    private func applyLateralInhibition(values: [Double]) -> [Double] {
        let count = values.count
        guard count > 0 else { return values }
        let mean = values.reduce(0.0, +) / Double(count)
        let strength = NerveBundleDefaults.lateralInhibitionStrength
        return values.map { $0 - strength * mean }
    }

    private func buildQuality(
        values: [Double],
        previousValues: [Double],
        timestamps: [TimeInterval],
        present: [Bool],
        time: TimeInterval
    ) -> [Double] {
        let count = values.count
        var quality = Array(repeating: NerveBundleDefaults.qualityFloor, count: count)
        for idx in 0..<count {
            var score = 1.0
            let delay = max(0.0, time - timestamps[idx])
            score -= NerveBundleDefaults.delayPenaltyPerSecond * delay
            if !present[idx] {
                score *= NerveBundleDefaults.missingPenalty
            }
            let delta = abs(values[idx] - previousValues[idx])
            score -= NerveBundleDefaults.deltaPenalty * min(1.0, delta)
            quality[idx] = clamp(score, min: NerveBundleDefaults.qualityFloor, max: 1.0)
        }
        return quality
    }

    private func alpha(for tau: TimeInterval, dt: TimeInterval) -> Double {
        guard dt > 0 else { return 1.0 }
        return dt / (tau + dt)
    }

    private func clamp(_ value: Double, min: Double, max: Double) -> Double {
        Swift.min(Swift.max(value, min), max)
    }
}
