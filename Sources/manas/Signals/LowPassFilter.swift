import Foundation

public struct LowPassFilter: Sendable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositiveDelta
        case nonPositiveCutoff
    }

    private let cutoffHz: Double
    private var lastOutput: Double
    private var hasState: Bool

    public init(cutoffHz: Double, initialValue: Double = 0.0) throws {
        guard cutoffHz.isFinite else { throw ValidationError.nonFinite("cutoffHz") }
        guard cutoffHz > 0 else { throw ValidationError.nonPositiveCutoff }
        guard initialValue.isFinite else { throw ValidationError.nonFinite("initialValue") }

        self.cutoffHz = cutoffHz
        self.lastOutput = initialValue
        self.hasState = false
    }

    public mutating func apply(input: Double, deltaTime: TimeInterval) throws -> Double {
        guard deltaTime.isFinite else { throw ValidationError.nonFinite("deltaTime") }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        if !hasState {
            lastOutput = input
            hasState = true
            return input
        }

        let rc = 1.0 / (2.0 * Double.pi * cutoffHz)
        let alpha = deltaTime / (rc + deltaTime)
        lastOutput = lastOutput + alpha * (input - lastOutput)
        return lastOutput
    }
}
