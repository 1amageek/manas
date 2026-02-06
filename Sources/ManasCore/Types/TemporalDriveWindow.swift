/// A time-windowed sequence of DriveIntents representing a multi-step action plan.
///
/// Inspired by Micro-World's temporal action windows, this enables controllers
/// to output a planning horizon rather than a single-step command.
public struct TemporalDriveWindow: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case emptyHorizon
        case nonPositiveInterval
        case nonFiniteInterval
    }

    /// Ordered sequence of drive intents over the planning horizon.
    public let horizon: [DriveIntent]

    /// Time interval between consecutive horizon steps, in seconds.
    public let intervalSeconds: Double

    /// Number of steps in the horizon.
    public var stepCount: Int { horizon.count }

    /// Total duration of the planning window, in seconds.
    public var durationSeconds: Double { Double(horizon.count) * intervalSeconds }

    /// The immediate (current-step) drive intent.
    public var current: DriveIntent { horizon[0] }

    public init(horizon: [DriveIntent], intervalSeconds: Double) throws {
        guard !horizon.isEmpty else {
            throw ValidationError.emptyHorizon
        }
        guard intervalSeconds.isFinite else {
            throw ValidationError.nonFiniteInterval
        }
        guard intervalSeconds > 0 else {
            throw ValidationError.nonPositiveInterval
        }
        self.horizon = horizon
        self.intervalSeconds = intervalSeconds
    }
}
