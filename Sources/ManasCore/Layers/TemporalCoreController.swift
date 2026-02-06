import Foundation

/// A core controller that outputs multi-step planning horizons.
///
/// Unlike `CoreController` which produces a single-step `[DriveIntent]`,
/// `TemporalCoreController` outputs `[TemporalDriveWindow]` — one window
/// per drive primitive, each containing a sequence of future intents.
public protocol TemporalCoreController {
    /// Produce multi-step drive windows for each drive primitive.
    ///
    /// - Parameters:
    ///   - trunks: Abstract feature streams from the neural hierarchy.
    ///   - time: Current simulation or real time.
    ///   - horizonSteps: Number of future steps to plan.
    /// - Returns: One `TemporalDriveWindow` per drive primitive.
    mutating func updateTemporal(
        trunks: TrunkBundle,
        time: TimeInterval,
        horizonSteps: Int
    ) throws -> [TemporalDriveWindow]
}
