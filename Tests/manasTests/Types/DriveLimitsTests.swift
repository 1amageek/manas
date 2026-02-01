import Testing
@testable import manas

@Test func driveLimitsRejectsInvalidRange() async throws {
    do {
        _ = try DriveLimits(limits: [DriveIndex(0): -Double.infinity...Double.infinity])
        #expect(Bool(false))
    } catch let error as DriveLimits.ValidationError {
        #expect(error == .nonFiniteRange(DriveIndex(0)))
    }
}

@Test func driveLimitsDetectsMissingLimit() async throws {
    let limits = try DriveLimits(limits: [DriveIndex(0): -1.0...1.0])
    let intent = try DriveIntent(index: DriveIndex(1), activation: 0.0)

    do {
        try limits.validate(intent)
        #expect(Bool(false))
    } catch let error as DriveLimits.ValidationError {
        #expect(error == .missingLimit(DriveIndex(1)))
    }
}

@Test func driveLimitsClampsActivation() async throws {
    let limits = try DriveLimits(limits: [DriveIndex(2): -1.0...1.0])
    let intent = try DriveIntent(index: DriveIndex(2), activation: 2.0)
    let clamped = try limits.clamped(intent)
    #expect(clamped.activation == 1.0)
}
