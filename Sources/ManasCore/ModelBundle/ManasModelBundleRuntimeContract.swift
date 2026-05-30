public struct ManasModelBundleRuntimeContract: Codable, Sendable, Equatable {
    public let embodimentHash: String
    public let configHash: String
    public let observationSchemaID: String
    public let driveSemanticsID: String
    public let motorNerveProfileID: String?
    public let corePeriodSeconds: Double?
    public let reflexPeriodSeconds: Double?

    public init(
        embodimentHash: String,
        configHash: String,
        observationSchemaID: String,
        driveSemanticsID: String,
        motorNerveProfileID: String? = nil,
        corePeriodSeconds: Double? = nil,
        reflexPeriodSeconds: Double? = nil
    ) {
        self.embodimentHash = embodimentHash
        self.configHash = configHash
        self.observationSchemaID = observationSchemaID
        self.driveSemanticsID = driveSemanticsID
        self.motorNerveProfileID = motorNerveProfileID
        self.corePeriodSeconds = corePeriodSeconds
        self.reflexPeriodSeconds = reflexPeriodSeconds
    }
}
