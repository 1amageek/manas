import Foundation

enum NerveBundleDefaults {
    static let qualityFloor: Double = 0.2
    static let transductionGain: Double = 2.0
    static let slowTau: TimeInterval = 0.05
    static let fastTau: TimeInterval = 0.005
    static let normalizationTau: TimeInterval = 0.2
    static let lateralInhibitionStrength: Double = 0.2
    static let delayPenaltyPerSecond: Double = 0.2
    static let missingPenalty: Double = 0.5
    static let deltaPenalty: Double = 0.1
    static let normalizationEpsilon: Double = 1.0e-6
}
