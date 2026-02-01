import Testing
@testable import manas

@Test func driveResidualsSubtractSaturationAndRate() async throws {
    let limits = try DriveLimits(limits: [DriveIndex(0): -1.0...1.0])
    let rates = try DriveRateLimits(limits: [DriveIndex(0): 1.0])
    let model = DriveResidualModel(driveLimits: limits, rateLimits: rates)

    let series: [[DriveIntent]] = [
        [try DriveIntent(index: DriveIndex(0), activation: 2.0)],
        [try DriveIntent(index: DriveIndex(0), activation: -1.0)],
    ]

    let residuals = try model.residuals(series: series, deltaTime: 1.0)
    #expect(residuals.count == 2)
    #expect(residuals[0][0] == 1.0)
    #expect(residuals[1][0] == -1.0)
}

