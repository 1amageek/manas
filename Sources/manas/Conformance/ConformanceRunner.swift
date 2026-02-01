import Foundation

public struct ConformanceRunner<Target: ManasConformanceTarget>: Sendable {
    public enum ValidationError: Error, Equatable {
        case frameCountMismatch
        case nonFiniteDelta
        case nonPositiveDelta
    }

    public var target: Target
    public let normalization: NormalizationBundle

    public init(target: Target, normalization: NormalizationBundle) {
        self.target = target
        self.normalization = normalization
    }

    public mutating func run(family: inout InputFamily) throws -> ConformanceRun {
        let frames = try family.frames()
        target.reset()

        var outputs: [[DriveIntent]] = []
        outputs.reserveCapacity(frames.count)

        for frame in frames {
            let energies = try frame.energies.map { sample in
                try EnergyState(index: sample.index, value: sample.value)
            }
            let phases = try frame.phases.map { sample in
                try PhaseState(index: sample.index, value: sample.value)
            }

            let drives = try target.step(
                energies: energies,
                phases: phases,
                deltaTime: family.deltaTime
            )
            outputs.append(drives)
        }

        return ConformanceRun(inputs: frames, outputs: outputs)
    }

    public func continuity(
        base: ConformanceRun,
        perturbed: ConformanceRun,
        l2: Double,
        lInf: Double
    ) throws -> [ContinuityCheck.Result] {
        guard base.inputs.count == perturbed.inputs.count else {
            throw ValidationError.frameCountMismatch
        }

        var results: [ContinuityCheck.Result] = []
        results.reserveCapacity(base.inputs.count)

        for index in base.inputs.indices {
            let baseFrame = base.inputs[index]
            let perturbedFrame = perturbed.inputs[index]

            let baseEnergies = try baseFrame.energies.map {
                try EnergyState(index: $0.index, value: $0.value)
            }
            let perturbedEnergies = try perturbedFrame.energies.map {
                try EnergyState(index: $0.index, value: $0.value)
            }
            let basePhases = try baseFrame.phases.map {
                try PhaseState(index: $0.index, value: $0.value)
            }
            let perturbedPhases = try perturbedFrame.phases.map {
                try PhaseState(index: $0.index, value: $0.value)
            }

            let baseInput = try normalization.normalizedInput(
                energies: baseEnergies,
                phases: basePhases
            )
            let perturbedInput = try normalization.normalizedInput(
                energies: perturbedEnergies,
                phases: perturbedPhases
            )

            let baseOutput = try normalization.normalizedOutput(
                drives: base.outputs[index]
            )
            let perturbedOutput = try normalization.normalizedOutput(
                drives: perturbed.outputs[index]
            )

            results.append(
                ContinuityCheck.evaluate(
                    input: baseInput,
                    inputPrime: perturbedInput,
                    output: baseOutput,
                    outputPrime: perturbedOutput,
                    l2: l2,
                    lInf: lInf
                )
            )
        }

        return results
    }

    public func totalVariation(
        run: ConformanceRun,
        limit: Double
    ) throws -> TotalVariationCheck.Result {
        var series: [[Double]] = []
        series.reserveCapacity(run.outputs.count)

        for drives in run.outputs {
            let normalized = try normalization.normalizedOutput(drives: drives)
            series.append(normalized)
        }

        return TotalVariationCheck.evaluate(series: series, limit: limit)
    }
}

