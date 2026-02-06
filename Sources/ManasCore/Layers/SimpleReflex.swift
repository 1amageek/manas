import Foundation

public struct SimpleReflex: ReflexController {
    public struct Configuration: Sendable, Equatable {
        public let driveCount: Int
        public let clampSensitivity: Double
        public let dampingSensitivity: Double
        public let deltaGain: Double
        public let deltaRange: ClosedRange<Double>

        public init(
            driveCount: Int,
            clampSensitivity: Double,
            dampingSensitivity: Double,
            deltaGain: Double,
            deltaRange: ClosedRange<Double> = -1.0...1.0
        ) {
            self.driveCount = max(0, driveCount)
            self.clampSensitivity = clampSensitivity
            self.dampingSensitivity = dampingSensitivity
            self.deltaGain = deltaGain
            self.deltaRange = deltaRange
        }
    }

    public var configuration: Configuration

    public init(configuration: Configuration) {
        self.configuration = configuration
    }

    public mutating func update(
        bundle: NerveBundleOutput,
        trunks: TrunkBundle,
        time: TimeInterval
    ) throws -> [ReflexCorrection] {
        _ = time
        _ = trunks
        var corrections: [ReflexCorrection] = []
        corrections.reserveCapacity(configuration.driveCount)

        for index in 0..<configuration.driveCount {
            let tap = index < bundle.fastTaps.count ? bundle.fastTaps[index] : 0.0
            let clamp = max(0.0, min(1.0, 1.0 - abs(tap) * configuration.clampSensitivity))
            let damping = max(0.0, min(1.0, abs(tap) * configuration.dampingSensitivity))
            let rawDelta = -tap * configuration.deltaGain
            let delta = min(max(rawDelta, configuration.deltaRange.lowerBound), configuration.deltaRange.upperBound)
            corrections.append(
                try ReflexCorrection(
                    driveIndex: DriveIndex(UInt32(index)),
                    clampMultiplier: clamp,
                    damping: damping,
                    delta: delta
                )
            )
        }

        return corrections
    }
}
