import Testing
@testable import ManasCore

private struct DriveScaleStage: MotorNerveStage {
    let scale: Double

    func map(
        input: [DriveIntent],
        telemetry: MotorNerveTelemetry
    ) throws -> [DriveIntent] {
        _ = telemetry
        return try input.map { drive in
            try DriveIntent(
                index: drive.index,
                activation: drive.activation * scale,
                parameters: drive.parameters
            )
        }
    }
}

@Test func motorNervePipelineChainsStages() async throws {
    let pipeline = MotorNervePipeline(
        first: DriveScaleStage(scale: 2.0),
        second: IdentityMotorNerveMapper()
    )
    let drives = [try DriveIntent(index: DriveIndex(0), activation: 0.25)]
    let outputs = try pipeline.map(input: drives, telemetry: MotorNerveTelemetry(motors: []))

    #expect(outputs.count == 1)
    #expect(abs(outputs[0].value - 0.5) < 1e-9)
}
