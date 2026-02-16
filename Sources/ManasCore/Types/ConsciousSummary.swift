public struct ConsciousSummary: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case invalidRange(String)
    }

    public let salience: Double
    public let risk: Double
    public let uncertainty: Double
    public let constraintPressure: Double
    public let recoveryState: Double
    public let timestamp: Double

    public init(
        salience: Double,
        risk: Double,
        uncertainty: Double,
        constraintPressure: Double,
        recoveryState: Double,
        timestamp: Double
    ) throws {
        guard salience.isFinite else {
            throw ValidationError.nonFinite("salience")
        }
        guard risk.isFinite else {
            throw ValidationError.nonFinite("risk")
        }
        guard uncertainty.isFinite else {
            throw ValidationError.nonFinite("uncertainty")
        }
        guard constraintPressure.isFinite else {
            throw ValidationError.nonFinite("constraintPressure")
        }
        guard recoveryState.isFinite else {
            throw ValidationError.nonFinite("recoveryState")
        }
        guard timestamp.isFinite else {
            throw ValidationError.nonFinite("timestamp")
        }
        if timestamp < 0 {
            throw ValidationError.invalidRange("timestamp")
        }

        self.salience = salience
        self.risk = risk
        self.uncertainty = uncertainty
        self.constraintPressure = constraintPressure
        self.recoveryState = recoveryState
        self.timestamp = timestamp
    }
}
