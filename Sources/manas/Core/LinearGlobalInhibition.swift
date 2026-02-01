public struct LinearGlobalInhibition: GlobalInhibitionModel {
    public enum ValidationError: Error, Equatable {
        case nonFiniteInput
        case nonFiniteThreshold
    }

    public init() {}

    public func factor(totalEnergy: Double, existThreshold: Double) throws -> Double {
        guard totalEnergy.isFinite else { throw ValidationError.nonFiniteInput }
        guard existThreshold.isFinite else { throw ValidationError.nonFiniteThreshold }

        guard existThreshold > 0 else {
            return 0.0
        }

        let ratio = totalEnergy / existThreshold
        let factor = max(0.0, 1.0 - ratio)
        return min(1.0, factor)
    }
}

