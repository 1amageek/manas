import Foundation
import MLX
import ManasCore

public struct ManasTrainingDatasetMetadata: Sendable, Codable, Equatable {
    public let scenarioId: String
    public let seed: UInt64
    public let timeStep: Double
    public let determinismTier: String
    public let configHash: String
    public let channelCount: Int
    public let driveCount: Int
    public let recordCount: Int
}

public struct ManasTrainingSensorSample: Sendable, Codable, Equatable {
    public let channelIndex: UInt32
    public let value: Double
    public let timestamp: Double
}

public struct ManasTrainingDriveIntent: Sendable, Codable, Equatable {
    public let driveIndex: UInt32
    public let value: Double
    public let parameters: [Double]

    public init(driveIndex: UInt32, value: Double, parameters: [Double] = []) {
        self.driveIndex = driveIndex
        self.value = value
        self.parameters = parameters
    }
}

public struct ManasTrainingReflexCorrection: Sendable, Codable, Equatable {
    public let driveIndex: UInt32
    public let clamp: Double
    public let damping: Double
    public let delta: Double
}

public struct ManasTrainingDatasetRecord: Sendable, Codable, Equatable {
    public let time: Double
    public let sensors: [ManasTrainingSensorSample]
    public let driveIntents: [ManasTrainingDriveIntent]
    public let reflexCorrections: [ManasTrainingReflexCorrection]
}

public struct ManasTrainingDataset {
    public let metadata: ManasTrainingDatasetMetadata
    public let records: [ManasTrainingDatasetRecord]

    public init(metadata: ManasTrainingDatasetMetadata, records: [ManasTrainingDatasetRecord]) {
        self.metadata = metadata
        self.records = records
    }

    public static func load(from directory: URL) throws -> ManasTrainingDataset {
        let metaURL = directory.appendingPathComponent("meta.json")
        let recordsURL = directory.appendingPathComponent("records.jsonl")
        let decoder = JSONDecoder()

        let metaData = try Data(contentsOf: metaURL)
        let metadata = try decoder.decode(ManasTrainingDatasetMetadata.self, from: metaData)

        let recordData = try String(contentsOf: recordsURL, encoding: .utf8)
        let lines = recordData.split(separator: "\n").filter { !$0.isEmpty }
        let records: [ManasTrainingDatasetRecord] = try lines.map { line in
            try decoder.decode(ManasTrainingDatasetRecord.self, from: Data(line.utf8))
        }

        return ManasTrainingDataset(metadata: metadata, records: records)
    }
}

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

public struct ManasTrainingBatchBuilder {
    public enum ValidationError: Error, Equatable {
        case invalidSequenceLength
        case driveCountMismatch(expected: Int, actual: Int)
    }

    public let sequenceLength: Int
    public let driveCount: Int
    public let maxBatches: Int?

    public init(sequenceLength: Int, driveCount: Int, maxBatches: Int? = nil) {
        self.sequenceLength = sequenceLength
        self.driveCount = driveCount
        self.maxBatches = maxBatches
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

        for start in 0...(vectors.count - sequenceLength) {
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

        for start in 0...(vectors.count - sequenceLength - 1) {
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
