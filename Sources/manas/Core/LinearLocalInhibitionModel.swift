public struct LinearLocalInhibitionModel: LocalInhibitionModel {
    public enum ValidationError: Error, Equatable {
        case nonFiniteInput
        case nonFiniteGradient
    }

    public init() {}

    public func factor(
        energy: Double,
        gradient: Double,
        thresholds: PerceptionThresholds
    ) throws -> Double {
        guard energy.isFinite else { throw ValidationError.nonFiniteInput }
        guard gradient.isFinite else { throw ValidationError.nonFiniteGradient }

        let energyFactor = LinearLocalInhibitionModel.scale(
            value: energy,
            start: thresholds.localEnergy,
            end: thresholds.reflexEnergy
        )

        let gradientFactor = LinearLocalInhibitionModel.scale(
            value: gradient,
            start: thresholds.criticalGradient,
            end: thresholds.reflexGradient
        )

        return min(energyFactor, gradientFactor)
    }

    private static func scale(value: Double, start: Double, end: Double) -> Double {
        guard end > start else {
            return value <= start ? 1.0 : 0.0
        }

        if value <= start {
            return 1.0
        }
        if value >= end {
            return 0.0
        }
        let ratio = (value - start) / (end - start)
        return max(0.0, min(1.0, 1.0 - ratio))
    }
}

