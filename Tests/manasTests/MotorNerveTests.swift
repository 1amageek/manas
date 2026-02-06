import Testing
@testable import ManasCore

@Test func motorNerveAppliesReflexCorrections() async throws {
    var motorNerve = ManasMotorNerve(
        mapper: IdentityMotorNerveMapper(),
        primitiveBank: try PrimitiveBankDescriptor(driveCount: 1)
    )

    let drives = [try DriveIntent(index: DriveIndex(0), activation: 0.8)]
    let corrections = [
        try ReflexCorrection(
            driveIndex: DriveIndex(0),
            clampMultiplier: 0.5,
            damping: 0.25,
            delta: 0.1
        )
    ]
    let telemetry = MotorNerveTelemetry(motors: [])
    let commands = try motorNerve.apply(
        drives: drives,
        corrections: corrections,
        telemetry: telemetry,
        deltaTime: 0.01
    )

    #expect(commands.count == 1)
    #expect(commands[0].value < 0.8)
    #expect(commands[0].value > 0.0)
}

@Test func motorNerveClampsDriveActivationToPrimitiveBounds() async throws {
    let descriptor = try PrimitiveDescriptor(id: DriveIndex(0), activationRange: -0.5...0.5)
    let bank = try PrimitiveBankDescriptor(driveCount: 1, primitives: [descriptor], defaultActivationRange: -1.0...1.0)
    var motorNerve = ManasMotorNerve(mapper: IdentityMotorNerveMapper(), primitiveBank: bank)

    let drives = [try DriveIntent(index: DriveIndex(0), activation: 1.2)]
    let telemetry = MotorNerveTelemetry(motors: [])
    let values = try motorNerve.apply(
        drives: drives,
        corrections: [],
        telemetry: telemetry,
        deltaTime: 0.01
    )

    #expect(values.count == 1)
    #expect(abs(values[0].value - 0.5) < 1e-9)
}

@Test func motorNerveRejectsOutOfRangeCorrectionIndex() async throws {
    var motorNerve = ManasMotorNerve(
        mapper: IdentityMotorNerveMapper(),
        primitiveBank: try PrimitiveBankDescriptor(driveCount: 2)
    )
    let drives = [
        try DriveIntent(index: DriveIndex(0), activation: 0.1),
        try DriveIntent(index: DriveIndex(1), activation: 0.2)
    ]
    let corrections = [
        try ReflexCorrection(
            driveIndex: DriveIndex(2),
            clampMultiplier: 1.0,
            damping: 0.0,
            delta: 0.0
        )
    ]
    let telemetry = MotorNerveTelemetry(motors: [])

    #expect(throws: ManasMotorNerve<IdentityMotorNerveMapper>.ValidationError.outOfRangeCorrection(DriveIndex(2))) {
        _ = try motorNerve.apply(
            drives: drives,
            corrections: corrections,
            telemetry: telemetry,
            deltaTime: 0.01
        )
    }
}
