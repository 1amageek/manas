import Foundation

public struct SimpleCore: CoreController {
    public struct Configuration: Sendable, Equatable {
        public let driveCount: Int
        public let activationRange: ClosedRange<Double>

        public init(driveCount: Int, activationRange: ClosedRange<Double>) {
            self.driveCount = max(0, driveCount)
            self.activationRange = activationRange
        }
    }

    public var configuration: Configuration

    public init(configuration: Configuration) {
        self.configuration = configuration
    }

    public mutating func update(trunks: TrunkBundle, time: TimeInterval) throws -> [DriveIntent] {
        _ = time
        var intents: [DriveIntent] = []
        intents.reserveCapacity(configuration.driveCount)

        for index in 0..<configuration.driveCount {
            let value = index < trunks.phase.count ? trunks.phase[index] : 0.0
            let clamped = min(max(value, configuration.activationRange.lowerBound), configuration.activationRange.upperBound)
            intents.append(try DriveIntent(index: DriveIndex(UInt32(index)), activation: clamped))
        }

        return intents
    }
}
