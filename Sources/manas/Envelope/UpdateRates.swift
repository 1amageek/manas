import Foundation

public struct UpdateRates: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(String)
        case nonPositive(String)
    }

    public let controllerUpdate: TimeInterval
    public let sensorSample: TimeInterval
    public let actuatorUpdate: TimeInterval

    public init(
        controllerUpdate: TimeInterval,
        sensorSample: TimeInterval,
        actuatorUpdate: TimeInterval
    ) throws {
        try UpdateRates.validatePositiveFinite(controllerUpdate, field: "controllerUpdate")
        try UpdateRates.validatePositiveFinite(sensorSample, field: "sensorSample")
        try UpdateRates.validatePositiveFinite(actuatorUpdate, field: "actuatorUpdate")

        self.controllerUpdate = controllerUpdate
        self.sensorSample = sensorSample
        self.actuatorUpdate = actuatorUpdate
    }

    private static func validatePositiveFinite(_ value: Double, field: String) throws {
        guard value.isFinite else { throw ValidationError.nonFinite(field) }
        guard value > 0 else { throw ValidationError.nonPositive(field) }
    }
}

