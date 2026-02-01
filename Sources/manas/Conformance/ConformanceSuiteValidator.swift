import Foundation

public struct ConformanceSuiteValidator {
    public enum ValidationError: Error, Equatable {
        case deltaTimeMismatch(TimeInterval, TimeInterval)
        case nonFiniteTolerance
        case negativeTolerance
    }

    public static func validateDeltaTime(
        families: [InputFamily],
        updateRates: UpdateRates,
        tolerance: TimeInterval = 0.0
    ) throws {
        guard tolerance.isFinite else { throw ValidationError.nonFiniteTolerance }
        guard tolerance >= 0 else { throw ValidationError.negativeTolerance }

        for family in families {
            let expected = updateRates.controllerUpdate
            if abs(family.deltaTime - expected) > tolerance {
                throw ValidationError.deltaTimeMismatch(family.deltaTime, expected)
            }
        }
    }
}

