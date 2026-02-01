import Testing
@testable import manas

@Test func attitudeStabilizerProducesThreeDrives() async throws {
    let gains = try AttitudeStabilizer.Gains(kp: 1.0, kd: 0.5)
    let stabilizer = try AttitudeStabilizer(
        driveRoll: DriveIndex(0),
        drivePitch: DriveIndex(1),
        driveYaw: DriveIndex(2),
        maxTorque: 1.0,
        maxAngularRate: 10.0,
        mapping: IMU6Mapping(),
        gains: gains
    )

    let phases = [
        try PhaseState(index: PhaseIndex(0), value: 0.1),
        try PhaseState(index: PhaseIndex(1), value: -0.1),
        try PhaseState(index: PhaseIndex(2), value: 0.0),
        try PhaseState(index: PhaseIndex(3), value: 0.2),
        try PhaseState(index: PhaseIndex(4), value: -0.2),
        try PhaseState(index: PhaseIndex(5), value: 0.0),
    ]

    let drives = try stabilizer.synthesize(energies: [], phases: phases, regime: .normal)
    #expect(drives.count == 3)
}

