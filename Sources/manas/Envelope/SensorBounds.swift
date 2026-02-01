import Foundation

public struct SensorBounds: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case negative(String)
    }

    public let noiseAmplitude: Double
    public let biasDrift: Double
    public let delay: TimeInterval
    public let bandwidthHz: Double

    public init(
        noiseAmplitude: Double,
        biasDrift: Double,
        delay: TimeInterval,
        bandwidthHz: Double
    ) throws {
        try SensorBounds.validateNonNegativeFinite(noiseAmplitude, field: "noiseAmplitude")
        try SensorBounds.validateNonNegativeFinite(biasDrift, field: "biasDrift")
        try SensorBounds.validateNonNegativeFinite(delay, field: "delay")
        try SensorBounds.validateNonNegativeFinite(bandwidthHz, field: "bandwidthHz")

        self.noiseAmplitude = noiseAmplitude
        self.biasDrift = biasDrift
        self.delay = delay
        self.bandwidthHz = bandwidthHz
    }

    private static func validateNonNegativeFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value >= 0 else { throw ValidationError.negative(field) }
    }
}

