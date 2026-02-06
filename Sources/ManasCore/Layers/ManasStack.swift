import Foundation

public struct ManasTimingConfig: Sendable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite
        case nonPositive
        case reflexNotFaster
    }

    public let corePeriod: TimeInterval
    public let reflexPeriod: TimeInterval

    public init(corePeriod: TimeInterval = 0.01, reflexPeriod: TimeInterval = 0.001) {
        precondition(corePeriod.isFinite && reflexPeriod.isFinite, "Timing periods must be finite.")
        precondition(corePeriod > 0 && reflexPeriod > 0, "Timing periods must be positive.")
        precondition(reflexPeriod <= corePeriod, "Reflex period must be <= Core period.")
        self.corePeriod = corePeriod
        self.reflexPeriod = reflexPeriod
    }
}

public struct ManasStack<
    Bundle: NerveBundle,
    Gate: Gating,
    Trunks: TrunkBuilder,
    Core: CoreController,
    Reflex: ReflexController,
    Mapper: MotorNerveMapper
> {
    public var bundle: Bundle
    public var gate: Gate
    public var trunks: Trunks
    public var core: Core
    public var reflex: Reflex
    public var motorNerve: ManasMotorNerve<Mapper>
    public var timing: ManasTimingConfig

    private var lastCoreUpdateTime: TimeInterval?
    private var lastReflexUpdateTime: TimeInterval?
    private var lastDrives: [DriveIntent]
    private var lastCorrections: [ReflexCorrection]

    public init(
        bundle: Bundle,
        gate: Gate,
        trunks: Trunks,
        core: Core,
        reflex: Reflex,
        motorNerve: ManasMotorNerve<Mapper>,
        timing: ManasTimingConfig = ManasTimingConfig()
    ) {
        self.bundle = bundle
        self.gate = gate
        self.trunks = trunks
        self.core = core
        self.reflex = reflex
        self.motorNerve = motorNerve
        self.timing = timing
        self.lastCoreUpdateTime = nil
        self.lastReflexUpdateTime = nil
        self.lastDrives = []
        self.lastCorrections = []
    }

    public mutating func update(
        samples: [SignalSample],
        time: TimeInterval,
        telemetry: MotorNerveTelemetry,
        deltaTime: TimeInterval
    ) throws -> [ActuatorValue] {
        let bundled = try bundle.process(samples: samples, time: time)
        let gated = try gate.apply(bundle: bundled, time: time)
        let trunkBundle = try trunks.build(from: gated, time: time)
        if shouldUpdateCore(now: time) {
            lastDrives = try core.update(trunks: trunkBundle, time: time)
            lastCoreUpdateTime = time
        }
        if lastDrives.isEmpty {
            lastDrives = try core.update(trunks: trunkBundle, time: time)
            lastCoreUpdateTime = time
        }

        if shouldUpdateReflex(now: time) {
            lastCorrections = try reflex.update(bundle: bundled, trunks: trunkBundle, time: time)
            lastReflexUpdateTime = time
        }
        if lastCorrections.isEmpty {
            lastCorrections = try reflex.update(bundle: bundled, trunks: trunkBundle, time: time)
            lastReflexUpdateTime = time
        }
        return try motorNerve.apply(
            drives: lastDrives,
            corrections: lastCorrections,
            telemetry: telemetry,
            deltaTime: deltaTime
        )
    }

    private func shouldUpdateCore(now: TimeInterval) -> Bool {
        guard let last = lastCoreUpdateTime else { return true }
        guard now.isFinite else { return true }
        if now < last { return true }
        return (now - last) >= timing.corePeriod
    }

    private func shouldUpdateReflex(now: TimeInterval) -> Bool {
        guard let last = lastReflexUpdateTime else { return true }
        guard now.isFinite else { return true }
        if now < last { return true }
        return (now - last) >= timing.reflexPeriod
    }
}
