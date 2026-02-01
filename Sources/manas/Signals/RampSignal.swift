import Foundation

public struct RampSignal: SignalGenerator {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
    }

    public let startValue: Double
    public let slope: Double

    public init(startValue: Double, slope: Double) throws {
        guard startValue.isFinite else { throw ValidationError.nonFinite("startValue") }
        guard slope.isFinite else { throw ValidationError.nonFinite("slope") }

        self.startValue = startValue
        self.slope = slope
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        startValue + slope * time
    }
}

