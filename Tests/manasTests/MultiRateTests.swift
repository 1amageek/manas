import Foundation
import Testing
@testable import ManasCore

private struct CountingCore: CoreController {
    var updateCount: Int = 0

    mutating func update(trunks: TrunkBundle, time: TimeInterval) throws -> [DriveIntent] {
        _ = trunks
        _ = time
        updateCount += 1
        return [try DriveIntent(index: DriveIndex(0), activation: 0.1)]
    }
}

private struct CountingReflex: ReflexController {
    var updateCount: Int = 0

    mutating func update(
        bundle: NerveBundleOutput,
        trunks: TrunkBundle,
        time: TimeInterval
    ) throws -> [ReflexCorrection] {
        _ = bundle
        _ = trunks
        _ = time
        updateCount += 1
        return [try ReflexCorrection(driveIndex: DriveIndex(0), clampMultiplier: 1.0, damping: 0.0, delta: 0.0)]
    }
}

@Test func manasStackHonorsMultiRateSchedule() async throws {
    let motorNerve = ManasMotorNerve(
        mapper: IdentityMotorNerveMapper(),
        primitiveBank: try PrimitiveBankDescriptor(driveCount: 1)
    )
    let timing = ManasTimingConfig(corePeriod: 0.01, reflexPeriod: 0.002)

    var stack = ManasStack(
        bundle: PassThroughNerveBundle(configuration: .init(channelCount: 1)),
        gate: IdentityGating(),
        trunks: BasicTrunksBuilder(),
        core: CountingCore(),
        reflex: CountingReflex(),
        motorNerve: motorNerve,
        timing: timing
    )

    let telemetry = MotorNerveTelemetry(motors: [])
    let deltaTime = 0.002

    for index in 0..<6 {
        let time = Double(index) * deltaTime
        let sample = try SignalSample(channelIndex: 0, value: 0.25, timestamp: time)
        _ = try stack.update(samples: [sample], time: time, telemetry: telemetry, deltaTime: deltaTime)
    }

    #expect(stack.core.updateCount == 2)
    #expect(stack.reflex.updateCount == 6)
}

@Test func manasStackHoldsCoreBetweenUpdates() async throws {
    let motorNerve = ManasMotorNerve(
        mapper: IdentityMotorNerveMapper(),
        primitiveBank: try PrimitiveBankDescriptor(driveCount: 1)
    )
    let timing = ManasTimingConfig(corePeriod: 0.01, reflexPeriod: 0.002)

    var stack = ManasStack(
        bundle: PassThroughNerveBundle(configuration: .init(channelCount: 1)),
        gate: IdentityGating(),
        trunks: BasicTrunksBuilder(),
        core: CountingCore(),
        reflex: CountingReflex(),
        motorNerve: motorNerve,
        timing: timing
    )

    let telemetry = MotorNerveTelemetry(motors: [])
    let deltaTime = 0.004
    let times: [Double] = [0.0, 0.004, 0.008]

    for time in times {
        let sample = try SignalSample(channelIndex: 0, value: 0.25, timestamp: time)
        _ = try stack.update(samples: [sample], time: time, telemetry: telemetry, deltaTime: deltaTime)
    }

    #expect(stack.core.updateCount == 1)
    #expect(stack.reflex.updateCount == 3)
}
