import Foundation

public struct PhaseBandwidthCheck: Sendable {
    public struct Result: Sendable, Equatable {
        public let maxDerivativeByIndex: [PhaseIndex: Double]
        public let passes: Bool
    }

    public static func evaluate(
        phases: [PhaseIndex: [Double]],
        deltaTime: TimeInterval,
        bandwidthHz: Double
    ) throws -> Result {
        guard deltaTime.isFinite else {
            throw ValidationError.nonFiniteDelta
        }
        guard deltaTime > 0 else {
            throw ValidationError.nonPositiveDelta
        }
        guard bandwidthHz.isFinite else {
            throw ValidationError.nonFiniteBandwidth
        }
        guard bandwidthHz >= 0 else {
            throw ValidationError.negativeBandwidth
        }

        var maxDerivativeByIndex: [PhaseIndex: Double] = [:]
        var passes = true

        for (index, series) in phases {
            let maxDerivative = maximumDerivative(series: series, deltaTime: deltaTime)
            let amplitude = series.map { abs($0) }.max() ?? 0.0
            let bound = 2.0 * Double.pi * bandwidthHz * amplitude
            maxDerivativeByIndex[index] = maxDerivative

            if amplitude == 0 {
                if maxDerivative > 0 {
                    passes = false
                }
                continue
            }

            if maxDerivative > bound {
                passes = false
            }
        }

        return Result(maxDerivativeByIndex: maxDerivativeByIndex, passes: passes)
    }

    private static func maximumDerivative(series: [Double], deltaTime: TimeInterval) -> Double {
        guard series.count >= 2 else { return 0.0 }
        var maxValue: Double = 0.0
        for idx in 1..<series.count {
            let derivative = abs(series[idx] - series[idx - 1]) / deltaTime
            if derivative > maxValue {
                maxValue = derivative
            }
        }
        return maxValue
    }

    public enum ValidationError: Error, Equatable {
        case nonFiniteDelta
        case nonPositiveDelta
        case nonFiniteBandwidth
        case negativeBandwidth
    }
}

