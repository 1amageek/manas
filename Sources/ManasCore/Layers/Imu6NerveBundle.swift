import Foundation

public struct Imu6NerveBundle: NerveBundle {
    public struct Configuration: Sendable, Equatable {
        public let gyroRange: ClosedRange<Double>
        public let accelRange: ClosedRange<Double>

        public init(
            gyroRange: ClosedRange<Double>,
            accelRange: ClosedRange<Double>
        ) {
            self.gyroRange = gyroRange
            self.accelRange = accelRange
        }
    }

    public enum ValidationError: Error, Equatable {
        case nonFinite
        case futureTimestamp(TimeInterval)
        case duplicateChannel(UInt32)
        case outOfRangeChannel(UInt32)
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
        self.lastProcessTime = nil
        self.lastValues = Array(repeating: 0.0, count: 6)
        self.lastSampleTimes = Array(repeating: 0.0, count: 6)
        self.slowState = Array(repeating: 0.0, count: 6)
        self.fastLowpassState = Array(repeating: 0.0, count: 6)
        self.normState = Array(repeating: 1.0, count: 6)
    }

    public mutating func process(samples: [SignalSample], time: TimeInterval) throws -> NerveBundleOutput {
        guard time.isFinite else { throw ValidationError.nonFinite }

        let dt: Double
        if let lastTime = lastProcessTime, time >= lastTime {
            dt = time - lastTime
        } else {
            dt = 0.0
        }
        lastProcessTime = time

        var values = lastValues
        var present = Array(repeating: false, count: 6)
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
            guard idx >= 0, idx < 6 else { throw ValidationError.outOfRangeChannel(sample.channelIndex) }
            values[idx] = sample.value
            timestamps[idx] = sample.timestamp
            present[idx] = true
        }

        lastValues = values
        lastSampleTimes = timestamps

        var transduced: [Double] = Array(repeating: 0.0, count: 6)
        for idx in 0..<6 {
            let range = idx < 3 ? configuration.gyroRange : configuration.accelRange
            let scaled = normalize(value: values[idx], range: range)
            transduced[idx] = tanh(scaled * NerveBundleDefaults.transductionGain)
        }

        let inhibited = applyLateralInhibition(values: transduced)

        let alphaNorm = alpha(for: NerveBundleDefaults.normalizationTau, dt: dt)
        for idx in 0..<6 {
            let target = abs(inhibited[idx])
            normState[idx] = normState[idx] + alphaNorm * (target - normState[idx])
        }

        var normalized: [Double] = Array(repeating: 0.0, count: 6)
        for idx in 0..<6 {
            let denom = max(NerveBundleDefaults.normalizationEpsilon, normState[idx])
            normalized[idx] = clamp(inhibited[idx] / denom, min: -1.0, max: 1.0)
        }

        let alphaSlow = alpha(for: NerveBundleDefaults.slowTau, dt: dt)
        let alphaFast = alpha(for: NerveBundleDefaults.fastTau, dt: dt)
        for idx in 0..<6 {
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

    private func normalize(value: Double, range: ClosedRange<Double>) -> Double {
        let clamped = min(max(value, range.lowerBound), range.upperBound)
        let span = range.upperBound - range.lowerBound
        guard span > 0 else { return 0.0 }
        let scaled = (clamped - range.lowerBound) / span
        return (scaled * 2.0) - 1.0
    }

    private func applyLateralInhibition(values: [Double]) -> [Double] {
        var output = values
        let gyroMean = (values[0] + values[1] + values[2]) / 3.0
        let accelMean = (values[3] + values[4] + values[5]) / 3.0

        let strength = NerveBundleDefaults.lateralInhibitionStrength
        for idx in 0..<3 {
            output[idx] = values[idx] - strength * gyroMean
        }
        for idx in 3..<6 {
            output[idx] = values[idx] - strength * accelMean
        }
        return output
    }

    private func buildQuality(
        values: [Double],
        previousValues: [Double],
        timestamps: [TimeInterval],
        present: [Bool],
        time: TimeInterval
    ) -> [Double] {
        var quality = Array(repeating: NerveBundleDefaults.qualityFloor, count: 6)

        for idx in 0..<6 {
            let range = idx < 3 ? configuration.gyroRange : configuration.accelRange
            let maxAbs = max(abs(range.lowerBound), abs(range.upperBound))
            guard maxAbs > 0 else {
                quality[idx] = NerveBundleDefaults.qualityFloor
                continue
            }

            let absValue = abs(values[idx])
            let saturation = max(0.0, (absValue - maxAbs) / maxAbs)
            var score = 1.0 - saturation

            let delay = max(0.0, time - timestamps[idx])
            score -= NerveBundleDefaults.delayPenaltyPerSecond * delay

            if !present[idx] {
                score *= NerveBundleDefaults.missingPenalty
            }

            let delta = abs(values[idx] - previousValues[idx]) / maxAbs
            score -= NerveBundleDefaults.deltaPenalty * delta

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
