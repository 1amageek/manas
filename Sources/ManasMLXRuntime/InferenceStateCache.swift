import MLX
import ManasMLXModels

/// Caches GRU hidden states across inference steps to avoid redundant computation.
///
/// For GRU-based models (ManasMLXCore, ManasMLXGoalCore), the hidden state from
/// the previous step is fed as input to the next step. This cache stores that
/// state and provides reset/warmup functionality.
public struct InferenceStateCache {
    public var coreState: ManasMLXCoreState?
    public var stepCount: Int

    public init() {
        self.coreState = nil
        self.stepCount = 0
    }

    /// Update the cached state after a forward pass.
    public mutating func update(newState: ManasMLXCoreState) {
        coreState = newState
        stepCount += 1
    }

    /// Reset the cache, clearing stored state.
    public mutating func reset() {
        coreState = nil
        stepCount = 0
    }

    /// Whether the cache has a valid state for the next step.
    public var hasState: Bool {
        coreState != nil
    }
}
