import Foundation
import Testing
@testable import ManasCore

private enum ArbitrationTrace {
    nonisolated(unsafe) static var events: [String] = []
    nonisolated(unsafe) static var mappedDriveActivations: [Double] = []

    static func reset() {
        events = []
        mappedDriveActivations = []
    }
}

private struct RecordingBundle: NerveBundle {
    mutating func process(samples: [SignalSample], time: TimeInterval) throws -> NerveBundleOutput {
        _ = samples
        _ = time
        ArbitrationTrace.events.append("bundle")
        return NerveBundleOutput(features: [1.0], fastTaps: [1.0], quality: [1.0])
    }
}

private struct RecordingGate: Gating {
    mutating func apply(bundle: NerveBundleOutput, time: TimeInterval) throws -> GatedBundle {
        _ = bundle
        _ = time
        ArbitrationTrace.events.append("gate")
        return GatedBundle(features: [1.0], fastTaps: [1.0], quality: [1.0], gateFactors: [1.0])
    }
}

private struct RecordingTrunks: TrunkBuilder {
    mutating func build(from gated: GatedBundle, time: TimeInterval) throws -> TrunkBundle {
        _ = gated
        _ = time
        ArbitrationTrace.events.append("trunks")
        return TrunkBundle(energy: [1.0], phase: [0.0], quality: [1.0], spike: [0.0])
    }
}

private struct RecordingCore: CoreController {
    let activation: Double

    mutating func update(trunks: TrunkBundle, time: TimeInterval) throws -> [DriveIntent] {
        _ = trunks
        _ = time
        ArbitrationTrace.events.append("core")
        return [try DriveIntent(index: DriveIndex(0), activation: activation)]
    }
}

private struct RecordingReflex: ReflexController {
    mutating func update(
        bundle: NerveBundleOutput,
        trunks: TrunkBundle,
        time: TimeInterval
    ) throws -> [ReflexCorrection] {
        _ = bundle
        _ = trunks
        _ = time
        ArbitrationTrace.events.append("reflex")
        return [try ReflexCorrection(driveIndex: DriveIndex(0), clampMultiplier: 0.5, damping: 0.0, delta: 0.0)]
    }
}

private struct RecordingMapper: MotorNerveMapper {
    func map(input drives: [DriveIntent], telemetry: MotorNerveTelemetry) throws -> [ActuatorValue] {
        _ = telemetry
        ArbitrationTrace.events.append("mapper")
        ArbitrationTrace.mappedDriveActivations = drives.map(\.activation)
        return try drives.enumerated().map { offset, drive in
            try ActuatorValue(index: ActuatorIndex(UInt32(offset)), value: drive.activation)
        }
    }
}

@Suite(.serialized)
struct ArbitrationFlowTests {
    @Test func manasStackAppliesOrderedArbitrationFlow() async throws {
        ArbitrationTrace.reset()
        let motorNerve = ManasMotorNerve(
            mapper: RecordingMapper(),
            primitiveBank: try PrimitiveBankDescriptor(driveCount: 1)
        )

        var stack = ManasStack(
            bundle: RecordingBundle(),
            gate: RecordingGate(),
            trunks: RecordingTrunks(),
            core: RecordingCore(activation: 0.8),
            reflex: RecordingReflex(),
            motorNerve: motorNerve
        )

        let sample = try SignalSample(channelIndex: 0, value: 1.0, timestamp: 0.0)
        let telemetry = MotorNerveTelemetry(motors: [])
        _ = try stack.update(samples: [sample], time: 0.0, telemetry: telemetry, deltaTime: 0.01)

        #expect(ArbitrationTrace.events == ["bundle", "gate", "trunks", "core", "reflex", "mapper"])
    }

    @Test func manasStackMergesReflexBeforeMotorNerveMapping() async throws {
        ArbitrationTrace.reset()
        let motorNerve = ManasMotorNerve(
            mapper: RecordingMapper(),
            primitiveBank: try PrimitiveBankDescriptor(driveCount: 1)
        )

        var stack = ManasStack(
            bundle: RecordingBundle(),
            gate: RecordingGate(),
            trunks: RecordingTrunks(),
            core: RecordingCore(activation: 1.0),
            reflex: RecordingReflex(),
            motorNerve: motorNerve
        )

        let sample = try SignalSample(channelIndex: 0, value: 1.0, timestamp: 0.0)
        let telemetry = MotorNerveTelemetry(motors: [])
        let output = try stack.update(samples: [sample], time: 0.0, telemetry: telemetry, deltaTime: 0.01)

        #expect(ArbitrationTrace.mappedDriveActivations.count == 1)
        #expect(abs(ArbitrationTrace.mappedDriveActivations[0] - 0.5) < 1e-9)
        #expect(abs(output[0].value - 0.5) < 1e-9)
    }
}
