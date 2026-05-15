import Foundation
import MLX
import ManasCore
import ManasMLXModels
import ManasTrainingData

public struct ManasTrunkPipeline {
    public var bundle: any NerveBundle
    public var gate: any Gating
    public var trunks: any TrunkBuilder

    public init(bundle: any NerveBundle, gate: any Gating, trunks: any TrunkBuilder) {
        self.bundle = bundle
        self.gate = gate
        self.trunks = trunks
    }

    public mutating func step(samples: [SignalSample], time: TimeInterval) throws -> (TrunkBundle, NerveBundleOutput) {
        let output = try bundle.process(samples: samples, time: time)
        let gated = try gate.apply(bundle: output, time: time)
        let trunkBundle = try trunks.build(from: gated, time: time)
        return (trunkBundle, output)
    }
}

public struct ManasTrainingWindowSampler: Sendable, Equatable {
    public let maxBatches: Int?

    public init(maxBatches: Int? = nil) {
        self.maxBatches = maxBatches
    }

    public func selectedWindowStarts(maxStart: Int) -> [Int] {
        guard maxStart >= 0 else { return [] }
        let windowCount = maxStart + 1
        guard let maxBatches, maxBatches < windowCount else {
            return Array(0...maxStart)
        }
        guard maxBatches > 1 else {
            return [maxStart / 2]
        }

        var selected: [Int] = []
        selected.reserveCapacity(maxBatches)
        var seen: Set<Int> = []
        for index in 0..<maxBatches {
            let fraction = Double(index) / Double(maxBatches - 1)
            let start = Int((fraction * Double(maxStart)).rounded())
            if seen.insert(start).inserted {
                selected.append(start)
            }
        }

        if selected.count < maxBatches {
            for start in 0...maxStart where seen.insert(start).inserted {
                selected.append(start)
                if selected.count == maxBatches {
                    break
                }
            }
        }

        return selected.sorted()
    }
}

public struct ManasTrainingBatchBuilder {
    public enum ValidationError: Error, Equatable {
        case invalidSequenceLength
        case driveCountMismatch(expected: Int, actual: Int)
    }

    public let sequenceLength: Int
    public let driveCount: Int
    public let maxBatches: Int?
    private let windowSampler: ManasTrainingWindowSampler

    public init(sequenceLength: Int, driveCount: Int, maxBatches: Int? = nil) {
        self.sequenceLength = sequenceLength
        self.driveCount = driveCount
        self.maxBatches = maxBatches
        self.windowSampler = ManasTrainingWindowSampler(maxBatches: maxBatches)
    }

    public func makeCoreBatches(
        dataset: ManasTrainingDataset,
        pipeline: inout ManasTrunkPipeline
    ) throws -> [ManasMLXSequenceBatch] {
        try validate(dataset: dataset)
        let vectors = try buildTrunkVectors(dataset: dataset, pipeline: &pipeline)
        let targets = buildDriveTargets(dataset: dataset)

        guard vectors.count == targets.count else { return [] }
        guard vectors.count >= sequenceLength else { return [] }

        var batches: [ManasMLXSequenceBatch] = []
        let trunkSize = vectors[0].count

        for start in selectedWindowStarts(maxStart: vectors.count - sequenceLength) {
            let trunkWindow = Array(vectors[start..<start + sequenceLength])
            let targetWindow = Array(targets[start..<start + sequenceLength])

            let trunksArray = MLXArray(
                converting: trunkWindow.flatMap { $0 }.map(Double.init),
                [sequenceLength, trunkSize]
            )
            let targetArray = MLXArray(
                converting: targetWindow.flatMap { $0 }.map(Double.init),
                [sequenceLength, driveCount]
            )
            batches.append(ManasMLXSequenceBatch(trunks: trunksArray, targetDrives: targetArray))
            if let maxBatches, batches.count >= maxBatches {
                break
            }
        }

        return batches
    }

    public func makeAuxBatches(
        dataset: ManasTrainingDataset,
        pipeline: inout ManasTrunkPipeline
    ) throws -> [ManasMLXAuxSequenceBatch] {
        try validate(dataset: dataset)
        let vectors = try buildTrunkVectors(dataset: dataset, pipeline: &pipeline)
        let targets = buildDriveTargets(dataset: dataset)

        guard vectors.count == targets.count else { return [] }
        guard vectors.count > sequenceLength else { return [] }

        var batches: [ManasMLXAuxSequenceBatch] = []
        let trunkSize = vectors[0].count

        for start in selectedWindowStarts(maxStart: vectors.count - sequenceLength - 1) {
            let trunkWindow = Array(vectors[start..<start + sequenceLength])
            let targetWindow = Array(targets[start..<start + sequenceLength])
            let auxTarget = vectors[start + sequenceLength]

            let trunksArray = MLXArray(
                converting: trunkWindow.flatMap { $0 }.map(Double.init),
                [sequenceLength, trunkSize]
            )
            let targetArray = MLXArray(
                converting: targetWindow.flatMap { $0 }.map(Double.init),
                [sequenceLength, driveCount]
            )
            let auxArray = MLXArray(
                converting: auxTarget.map(Double.init),
                [1, trunkSize]
            )

            batches.append(ManasMLXAuxSequenceBatch(trunks: trunksArray, targetDrives: targetArray, targetAux: auxArray))
            if let maxBatches, batches.count >= maxBatches {
                break
            }
        }

        return batches
    }

    public func makeReflexBatches(
        dataset: ManasTrainingDataset,
        pipeline: inout ManasTrunkPipeline
    ) throws -> [ManasMLXReflexBatch] {
        try validate(dataset: dataset)
        let records = dataset.records
        var batches: [ManasMLXReflexBatch] = []

        for record in records {
            let samples = try record.sensors.map { sample in
                try SignalSample(channelIndex: sample.channelIndex, value: sample.value, timestamp: sample.timestamp)
            }
            let (_, output) = try pipeline.step(samples: samples, time: record.time)
            let input = MLXArray(
                converting: output.fastTaps,
                [1, output.fastTaps.count]
            )
            let targets = mapReflexTargets(record.reflexCorrections)

            let clamp = MLXArray(converting: targets.clamp.map(Double.init), [1, driveCount])
            let damping = MLXArray(converting: targets.damping.map(Double.init), [1, driveCount])
            let delta = MLXArray(converting: targets.delta.map(Double.init), [1, driveCount])
            batches.append(ManasMLXReflexBatch(inputs: input, targetClamp: clamp, targetDamping: damping, targetDelta: delta))
            if let maxBatches, batches.count >= maxBatches {
                break
            }
        }

        return batches
    }

    public func makeWorldModelBatches(
        dataset: ManasTrainingDataset,
        pipeline: inout ManasTrunkPipeline,
        rewardConfig: DenseRewardConfig = DenseRewardConfig()
    ) throws -> [ManasMLXWorldModelBatch] {
        try validate(dataset: dataset)
        let vectors = try buildTrunkVectors(dataset: dataset, pipeline: &pipeline)
        let targets = buildDriveTargets(dataset: dataset)
        let rewards = buildRewards(dataset: dataset, driveTargets: targets, rewardConfig: rewardConfig)

        guard vectors.count == targets.count else { return [] }
        guard vectors.count >= sequenceLength else { return [] }

        var batches: [ManasMLXWorldModelBatch] = []
        let trunkSize = vectors[0].count

        for start in selectedWindowStarts(maxStart: vectors.count - sequenceLength) {
            guard !crossesEpisodeBoundary(records: dataset.records, start: start, length: sequenceLength) else {
                continue
            }
            let trunkWindow = Array(vectors[start..<start + sequenceLength])
            let targetWindow = Array(targets[start..<start + sequenceLength])
            let rewardWindow = Array(rewards[start..<start + sequenceLength])

            let trunksArray = MLXArray(
                converting: trunkWindow.flatMap { $0 }.map(Double.init),
                [sequenceLength, trunkSize]
            )
            let targetArray = MLXArray(
                converting: targetWindow.flatMap { $0 }.map(Double.init),
                [sequenceLength, driveCount]
            )
            let rewardArray = MLXArray(
                converting: rewardWindow.map(Double.init),
                [sequenceLength, 1]
            )

            var continueValues = Array(repeating: Float(1.0), count: sequenceLength)
            let terminalRecord = dataset.records[start + sequenceLength - 1]
            if terminalRecord.done == true || terminalRecord.truncated == true || start + sequenceLength == vectors.count {
                continueValues[sequenceLength - 1] = 0.0
            }
            let continueArray = MLXArray(
                converting: continueValues.map(Double.init),
                [sequenceLength, 1]
            )

            batches.append(ManasMLXWorldModelBatch(
                trunks: trunksArray,
                targetDrives: targetArray,
                rewards: rewardArray,
                continues: continueArray
            ))
        }

        return batches
    }

    private func buildRewards(
        dataset: ManasTrainingDataset,
        driveTargets: [[Float]],
        rewardConfig: DenseRewardConfig
    ) -> [Float] {
        let records = dataset.records
        var rewards: [Float] = []
        rewards.reserveCapacity(records.count)

        for (index, record) in records.enumerated() {
            if let reward = record.reward {
                rewards.append(Float(reward))
                continue
            }
            let tilt = Float(record.tiltRadians ?? 0)
            let omega = Float(record.omegaMagnitude ?? 0)
            let activations = driveTargets[index]
            let previousActivations: [Float]? = index > 0 ? driveTargets[index - 1] : nil

            let reward = DenseRewardComputer.computeReward(
                tiltRadians: tilt,
                omegaMagnitude: omega,
                driveActivations: activations,
                previousDriveActivations: previousActivations,
                config: rewardConfig
            )
            rewards.append(reward)
        }

        return rewards
    }

    private func crossesEpisodeBoundary(
        records: [ManasTrainingDatasetRecord],
        start: Int,
        length: Int
    ) -> Bool {
        let window = records[start..<start + length]
        let episodeIds = Set(window.compactMap(\.episodeId))
        return episodeIds.count > 1
    }

    private func validate(dataset: ManasTrainingDataset) throws {
        guard sequenceLength > 0 else { throw ValidationError.invalidSequenceLength }
        let actual = dataset.metadata.driveCount
        guard actual == driveCount else { throw ValidationError.driveCountMismatch(expected: driveCount, actual: actual) }
    }

    private func buildTrunkVectors(
        dataset: ManasTrainingDataset,
        pipeline: inout ManasTrunkPipeline
    ) throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(dataset.records.count)

        for record in dataset.records {
            let samples = try record.sensors.map { sample in
                try SignalSample(channelIndex: sample.channelIndex, value: sample.value, timestamp: sample.timestamp)
            }
            let (trunks, _) = try pipeline.step(samples: samples, time: record.time)
            vectors.append(concatTrunks(trunks))
        }

        return vectors
    }

    private func buildDriveTargets(dataset: ManasTrainingDataset) -> [[Float]] {
        dataset.records.map { record in
            var drives = Array(repeating: Float(0), count: driveCount)
            for intent in record.driveIntents {
                let index = Int(intent.driveIndex)
                guard index >= 0 && index < driveCount else { continue }
                drives[index] = Float(intent.value)
            }
            return drives
        }
    }

    private func selectedWindowStarts(maxStart: Int) -> [Int] {
        windowSampler.selectedWindowStarts(maxStart: maxStart)
    }

    private func mapReflexTargets(_ corrections: [ManasTrainingReflexCorrection]) -> (clamp: [Float], damping: [Float], delta: [Float]) {
        var clamp = Array(repeating: Float(1.0), count: driveCount)
        var damping = Array(repeating: Float(0.0), count: driveCount)
        var delta = Array(repeating: Float(0.0), count: driveCount)

        for correction in corrections {
            let index = Int(correction.driveIndex)
            guard index >= 0 && index < driveCount else { continue }
            clamp[index] = Float(correction.clamp)
            damping[index] = Float(correction.damping)
            delta[index] = Float(correction.delta)
        }

        return (clamp, damping, delta)
    }

    private func concatTrunks(_ bundle: TrunkBundle) -> [Float] {
        bundle.energy.map(Float.init)
        + bundle.phase.map(Float.init)
        + bundle.quality.map(Float.init)
        + bundle.spike.map(Float.init)
    }
}
