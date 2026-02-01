import Foundation

public struct ManasCore<
    Synthesizer: DriveSynthesizer,
    GlobalInhibition: GlobalInhibitionModel,
    LocalInhibition: LocalInhibitionModel,
    Reflex: ReflexPolicy,
    GradientEstimator: EnergyGradientEstimator
>: Sendable {
    public var parameters: ManasParameters
    public var driveSynthesizer: Synthesizer
    public var globalInhibition: GlobalInhibition
    public var localInhibition: LocalInhibition
    public var coupling: InhibitionCoupling
    public var reflexPolicy: Reflex
    public var gradientEstimator: GradientEstimator

    public init(
        parameters: ManasParameters,
        driveSynthesizer: Synthesizer,
        globalInhibition: GlobalInhibition,
        localInhibition: LocalInhibition,
        coupling: InhibitionCoupling,
        reflexPolicy: Reflex,
        gradientEstimator: GradientEstimator
    ) {
        self.parameters = parameters
        self.driveSynthesizer = driveSynthesizer
        self.globalInhibition = globalInhibition
        self.localInhibition = localInhibition
        self.coupling = coupling
        self.reflexPolicy = reflexPolicy
        self.gradientEstimator = gradientEstimator
    }

    public mutating func update(
        energies: [EnergyState],
        phases: [PhaseState],
        deltaTime: TimeInterval
    ) throws -> ManasOutput {
        let totalEnergy = try computeTotalEnergy(energies)
        let regime: Regime = totalEnergy >= parameters.globalThresholds.existEnergy ? .survivalOnly : .normal
        let gradients = try gradientEstimator.estimate(energies: energies, deltaTime: deltaTime)

        let reflexes = computeReflexes(energies: energies, gradients: gradients)
        let baseDrives = try driveSynthesizer.synthesize(energies: energies, phases: phases, regime: regime)

        let modelFactor = try globalInhibition.factor(
            totalEnergy: totalEnergy,
            existThreshold: parameters.globalThresholds.existEnergy
        )
        let globalFactor = regime == .survivalOnly ? 0.0 : modelFactor
        guard globalFactor.isFinite, globalFactor >= 0, globalFactor <= 1 else {
            throw ManasCoreError.invalidGlobalInhibition(globalFactor)
        }

        var drives = try applyGlobalInhibition(baseDrives, factor: globalFactor)
        drives = try applyLocalInhibition(
            drives: drives,
            energies: energies,
            gradients: gradients
        )

        drives = try drives.map { try parameters.driveLimits.clamped($0) }

        let overrides = try reflexPolicy.overrides(
            reflexes: reflexes,
            regime: regime,
            baseDrives: drives
        )
        if !reflexes.isEmpty {
            guard let overrideDrives = overrides else {
                throw ManasCoreError.reflexNotHandled(reflexes)
            }
            drives = try overrideDrives.map { try parameters.driveLimits.clamped($0) }
        }

        return ManasOutput(
            regime: regime,
            totalEnergy: totalEnergy,
            globalInhibition: globalFactor,
            reflexes: reflexes,
            drives: drives
        )
    }

    public mutating func reset() {
        gradientEstimator.reset()
    }

    private func computeTotalEnergy(_ energies: [EnergyState]) throws -> Double {
        var total: Double = 0
        for energy in energies {
            guard let weight = parameters.weights.weight(for: energy.index) else {
                throw ManasCoreError.missingWeight(energy.index)
            }
            total += weight * energy.value
        }
        return total
    }

    private func computeReflexes(
        energies: [EnergyState],
        gradients: [EnergyGradient]
    ) -> [PerceptionIndex] {
        var gradientByIndex: [PerceptionIndex: Double] = [:]
        for gradient in gradients {
            gradientByIndex[gradient.index] = gradient.value
        }

        var reflexes: [PerceptionIndex] = []
        reflexes.reserveCapacity(energies.count)

        for energy in energies {
            guard let thresholds = parameters.thresholds[energy.index] else {
                continue
            }
            let gradient = gradientByIndex[energy.index] ?? 0.0
            if energy.value >= thresholds.reflexEnergy || gradient >= thresholds.reflexGradient {
                reflexes.append(energy.index)
            }
        }

        return reflexes
    }

    private func applyGlobalInhibition(
        _ drives: [DriveIntent],
        factor: Double
    ) throws -> [DriveIntent] {
        guard factor.isFinite, factor >= 0, factor <= 1 else {
            throw ManasCoreError.invalidGlobalInhibition(factor)
        }
        return try drives.map { drive in
            try DriveIntent(index: drive.index, activation: drive.activation * factor)
        }
    }

    private func applyLocalInhibition(
        drives: [DriveIntent],
        energies: [EnergyState],
        gradients: [EnergyGradient]
    ) throws -> [DriveIntent] {
        var gradientByIndex: [PerceptionIndex: Double] = [:]
        for gradient in gradients {
            gradientByIndex[gradient.index] = gradient.value
        }

        var localFactors: [PerceptionIndex: Double] = [:]
        for energy in energies {
            guard let thresholds = parameters.thresholds[energy.index] else {
                throw ManasCoreError.missingThreshold(energy.index)
            }
            let gradient = gradientByIndex[energy.index] ?? 0.0
            let factor = try localInhibition.factor(
                energy: energy.value,
                gradient: gradient,
                thresholds: thresholds
            )
            guard factor.isFinite, factor >= 0, factor <= 1 else {
                throw ManasCoreError.invalidInhibitionFactor(factor)
            }
            localFactors[energy.index] = factor
        }

        return try drives.map { drive in
            var factor: Double = 1.0
            for (perception, inhibitionFactor) in localFactors {
                let couplingValue = coupling.coupling(for: perception, drive: drive.index)
                let adjusted = 1.0 - couplingValue * (1.0 - inhibitionFactor)
                factor *= adjusted
            }

            guard factor.isFinite, factor >= 0 else {
                throw ManasCoreError.invalidInhibitionFactor(factor)
            }

            return try DriveIntent(index: drive.index, activation: drive.activation * factor)
        }
    }
}

extension ManasCore: ManasConformanceTarget {
    public mutating func step(
        energies: [EnergyState],
        phases: [PhaseState],
        deltaTime: TimeInterval
    ) throws -> [DriveIntent] {
        try update(energies: energies, phases: phases, deltaTime: deltaTime).drives
    }
}
