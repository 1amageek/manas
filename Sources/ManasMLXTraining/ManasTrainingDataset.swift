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
        guard maxBatches > 0 else {
            return []
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

    public func selectedWindowStartBatches(maxStart: Int, miniBatchSize: Int) -> [[Int]] {
        let starts = selectedWindowStarts(maxStart: maxStart)
        let resolvedMiniBatchSize = max(1, miniBatchSize)
        var chunks: [[Int]] = []
        chunks.reserveCapacity((starts.count + resolvedMiniBatchSize - 1) / resolvedMiniBatchSize)
        var index = 0
        while index < starts.count {
            let end = min(index + resolvedMiniBatchSize, starts.count)
            chunks.append(Array(starts[index..<end]))
            index = end
        }
        return chunks
    }
}

public struct ManasTrainingBatchBuilder {
    public static let defaultMiniBatchSize = 32

    public static func normalizedMiniBatchSize(_ miniBatchSize: Int?) -> Int {
        max(1, miniBatchSize ?? defaultMiniBatchSize)
    }

    public enum ValidationError: Error, Equatable {
        case invalidSequenceLength
        case driveCountMismatch(expected: Int, actual: Int)
    }

    public let sequenceLength: Int
    public let driveCount: Int
    public let maxBatches: Int?
    public let miniBatchSize: Int
    private let windowSampler: ManasTrainingWindowSampler

    public init(
        sequenceLength: Int,
        driveCount: Int,
        maxBatches: Int? = nil,
        miniBatchSize: Int = ManasTrainingBatchBuilder.defaultMiniBatchSize
    ) {
        self.sequenceLength = sequenceLength
        self.driveCount = driveCount
        self.maxBatches = maxBatches
        self.miniBatchSize = Self.normalizedMiniBatchSize(miniBatchSize)
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

        let trunkSize = vectors[0].count
        let starts = selectedWindowStarts(maxStart: vectors.count - sequenceLength)
        var batches: [ManasMLXSequenceBatch] = []
        batches.reserveCapacity((starts.count + miniBatchSize - 1) / miniBatchSize)

        for chunk in windowStartChunks(starts) {
            let trunkValues = ManasMLXOwnedFloatTensorBuffer(
                capacity: chunk.count * sequenceLength * trunkSize
            )
            let targetValues = ManasMLXOwnedFloatTensorBuffer(
                capacity: chunk.count * sequenceLength * driveCount
            )

            for start in chunk {
                appendWindowValues(from: vectors, start: start, length: sequenceLength, to: trunkValues)
                appendWindowValues(from: targets, start: start, length: sequenceLength, to: targetValues)
            }

            batches.append(ManasMLXSequenceBatch(
                trunks: trunkValues.makeArray(shape: [chunk.count, sequenceLength, trunkSize]),
                targetDrives: targetValues.makeArray(shape: [chunk.count, sequenceLength, driveCount])
            ))
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

        let trunkSize = vectors[0].count
        let starts = selectedWindowStarts(maxStart: vectors.count - sequenceLength - 1)
        var batches: [ManasMLXAuxSequenceBatch] = []
        batches.reserveCapacity((starts.count + miniBatchSize - 1) / miniBatchSize)

        for chunk in windowStartChunks(starts) {
            let trunkValues = ManasMLXOwnedFloatTensorBuffer(
                capacity: chunk.count * sequenceLength * trunkSize
            )
            let targetValues = ManasMLXOwnedFloatTensorBuffer(
                capacity: chunk.count * sequenceLength * driveCount
            )
            let auxValues = ManasMLXOwnedFloatTensorBuffer(capacity: chunk.count * trunkSize)

            for start in chunk {
                appendWindowValues(from: vectors, start: start, length: sequenceLength, to: trunkValues)
                appendWindowValues(from: targets, start: start, length: sequenceLength, to: targetValues)
                auxValues.append(contentsOf: vectors[start + sequenceLength])
            }

            batches.append(ManasMLXAuxSequenceBatch(
                trunks: trunkValues.makeArray(shape: [chunk.count, sequenceLength, trunkSize]),
                targetDrives: targetValues.makeArray(shape: [chunk.count, sequenceLength, driveCount]),
                targetAux: auxValues.makeArray(shape: [chunk.count, 1, trunkSize])
            ))
        }

        return batches
    }

    public func makeReflexBatches(
        dataset: ManasTrainingDataset,
        pipeline: inout ManasTrunkPipeline
    ) throws -> [ManasMLXReflexBatch] {
        try validate(dataset: dataset)
        if let maxBatches, maxBatches <= 0 { return [] }
        let records = dataset.records
        var batches: [ManasMLXReflexBatch] = []

        for record in records {
            let samples = try record.sensors.map { sample in
                try SignalSample(channelIndex: sample.channelIndex, value: sample.value, timestamp: sample.timestamp)
            }
            let (_, output) = try pipeline.step(samples: samples, time: record.time)
            let input = ManasMLXOwnedFloatTensorBuffer.makeArray(
                converting: output.fastTaps,
                shape: [1, output.fastTaps.count]
            )
            let targets = mapReflexTargets(record.reflexCorrections)

            let clamp = ManasMLXOwnedFloatTensorBuffer.makeArray(values: targets.clamp, shape: [1, driveCount])
            let damping = ManasMLXOwnedFloatTensorBuffer.makeArray(values: targets.damping, shape: [1, driveCount])
            let delta = ManasMLXOwnedFloatTensorBuffer.makeArray(values: targets.delta, shape: [1, driveCount])
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

        let trunkSize = vectors[0].count
        let starts = selectedWindowStarts(maxStart: vectors.count - sequenceLength)
            .filter { start in
                !crossesEpisodeBoundary(records: dataset.records, start: start, length: sequenceLength)
            }
        var batches: [ManasMLXWorldModelBatch] = []
        batches.reserveCapacity((starts.count + miniBatchSize - 1) / miniBatchSize)

        for chunk in windowStartChunks(starts) {
            let trunkValues = ManasMLXOwnedFloatTensorBuffer(
                capacity: chunk.count * sequenceLength * trunkSize
            )
            let targetValues = ManasMLXOwnedFloatTensorBuffer(
                capacity: chunk.count * sequenceLength * driveCount
            )
            let rewardValues = ManasMLXOwnedFloatTensorBuffer(capacity: chunk.count * sequenceLength)
            let continueValues = ManasMLXOwnedFloatTensorBuffer(capacity: chunk.count * sequenceLength)

            for start in chunk {
                appendWindowValues(from: vectors, start: start, length: sequenceLength, to: trunkValues)
                appendWindowValues(from: targets, start: start, length: sequenceLength, to: targetValues)
                appendScalarWindowValues(from: rewards, start: start, length: sequenceLength, to: rewardValues)

                let terminalRecord = dataset.records[start + sequenceLength - 1]
                let terminalStopsContinuation = terminalRecord.done == true
                    || terminalRecord.truncated == true
                    || start + sequenceLength == vectors.count
                for offset in 0..<sequenceLength {
                    continueValues.append(terminalStopsContinuation && offset == sequenceLength - 1 ? 0 : 1)
                }
            }

            batches.append(ManasMLXWorldModelBatch(
                trunks: trunkValues.makeArray(shape: [chunk.count, sequenceLength, trunkSize]),
                targetDrives: targetValues.makeArray(shape: [chunk.count, sequenceLength, driveCount]),
                rewards: rewardValues.makeArray(shape: [chunk.count, sequenceLength, 1]),
                continues: continueValues.makeArray(shape: [chunk.count, sequenceLength, 1])
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

    private func windowStartChunks(_ starts: [Int]) -> [[Int]] {
        var chunks: [[Int]] = []
        chunks.reserveCapacity((starts.count + miniBatchSize - 1) / miniBatchSize)
        var index = 0
        while index < starts.count {
            let end = min(index + miniBatchSize, starts.count)
            chunks.append(Array(starts[index..<end]))
            index = end
        }
        return chunks
    }

    private func appendWindowValues(
        from rows: [[Float]],
        start: Int,
        length: Int,
        to values: ManasMLXOwnedFloatTensorBuffer
    ) {
        for index in start..<start + length {
            values.append(contentsOf: rows[index])
        }
    }

    private func appendScalarWindowValues(
        from rows: [Float],
        start: Int,
        length: Int,
        to values: ManasMLXOwnedFloatTensorBuffer
    ) {
        for index in start..<start + length {
            values.append(rows[index])
        }
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
