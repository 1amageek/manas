import Foundation

public struct SignalSample: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case nonFinite
        case negativeTimestamp
    }

    public let channelIndex: UInt32
    public let value: Double
    public let timestamp: TimeInterval

    public init(channelIndex: UInt32, value: Double, timestamp: TimeInterval) throws {
        guard value.isFinite, timestamp.isFinite else { throw ValidationError.nonFinite }
        guard timestamp >= 0 else { throw ValidationError.negativeTimestamp }
        self.channelIndex = channelIndex
        self.value = value
        self.timestamp = timestamp
    }
}
