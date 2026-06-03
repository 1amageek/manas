import MLX
import ManasCore

final class ManasMLXRuntimeFloatTensorBuffer {
    let capacity: Int
    private(set) var count = 0

    private let pointer: UnsafeMutablePointer<Float>
    private var transferred = false

    init(capacity: Int) {
        precondition(capacity >= 0)
        self.capacity = capacity
        self.pointer = UnsafeMutablePointer<Float>.allocate(capacity: max(1, capacity))
    }

    deinit {
        guard !transferred else { return }
        pointer.deinitialize(count: count)
        pointer.deallocate()
    }

    func append(_ value: Float) {
        precondition(count < capacity)
        pointer.advanced(by: count).initialize(to: value)
        count += 1
    }

    func append(_ value: Double) {
        append(Float(value))
    }

    func append(contentsOf values: [Float]) {
        guard !values.isEmpty else { return }
        precondition(count + values.count <= capacity)
        values.withUnsafeBufferPointer { source in
            guard let baseAddress = source.baseAddress else { return }
            pointer.advanced(by: count).initialize(from: baseAddress, count: values.count)
        }
        count += values.count
    }

    func append(contentsOf values: [Double]) {
        guard !values.isEmpty else { return }
        precondition(count + values.count <= capacity)
        let start = count
        for offset in values.indices {
            pointer.advanced(by: start + offset).initialize(to: Float(values[offset]))
        }
        count += values.count
    }

    func appendZeros(count zeroCount: Int) {
        guard zeroCount > 0 else { return }
        precondition(count + zeroCount <= capacity)
        pointer.advanced(by: count).initialize(repeating: 0, count: zeroCount)
        count += zeroCount
    }

    func makeArray(shape: [Int]) -> MLXArray {
        precondition(count == capacity)
        precondition(shape.reduce(1, *) == capacity)
        transferred = true
        let pointer = self.pointer
        let capacity = self.capacity
        return MLXArray(rawPointer: UnsafeMutableRawPointer(pointer), shape, dtype: .float32) {
            pointer.deinitialize(count: capacity)
            pointer.deallocate()
        }
    }
}

enum ManasMLXRuntimeTensorInput {
    static func trunkInput(from trunks: TrunkBundle) -> MLXArray {
        let size = trunkDimension(trunks)
        let buffer = ManasMLXRuntimeFloatTensorBuffer(capacity: size)
        appendTrunks(trunks, to: buffer)
        return buffer.makeArray(shape: [1, 1, size])
    }

    static func trunkVector(from trunks: TrunkBundle) -> [Float] {
        var vector: [Float] = []
        vector.reserveCapacity(trunkDimension(trunks))
        appendTrunks(trunks, to: &vector)
        return vector
    }

    static func trunkHistoryInput(history: [[Float]], inputSize: Int) -> MLXArray {
        let sequenceLength = history.count
        let buffer = ManasMLXRuntimeFloatTensorBuffer(capacity: sequenceLength * inputSize)
        for row in history {
            precondition(row.count == inputSize)
            buffer.append(contentsOf: row)
        }
        return buffer.makeArray(shape: [1, sequenceLength, inputSize])
    }

    static func descendingInput(goals: [ControlGoal], size: Int) -> MLXArray {
        let buffer = ManasMLXRuntimeFloatTensorBuffer(capacity: size)
        appendGoalVector(goals: goals, size: size, to: buffer)
        return buffer.makeArray(shape: [1, size])
    }

    static func goalInput(goals: [ControlGoal], size: Int) -> MLXArray {
        let buffer = ManasMLXRuntimeFloatTensorBuffer(capacity: size)
        appendGoalVector(goals: goals, size: size, to: buffer)
        return buffer.makeArray(shape: [1, 1, size])
    }

    private static func trunkDimension(_ trunks: TrunkBundle) -> Int {
        trunks.energy.count + trunks.phase.count + trunks.quality.count + trunks.spike.count
    }

    private static func appendTrunks(_ trunks: TrunkBundle, to buffer: ManasMLXRuntimeFloatTensorBuffer) {
        append(trunks.energy, to: buffer)
        append(trunks.phase, to: buffer)
        append(trunks.quality, to: buffer)
        append(trunks.spike, to: buffer)
    }

    private static func appendTrunks(_ trunks: TrunkBundle, to vector: inout [Float]) {
        append(trunks.energy, to: &vector)
        append(trunks.phase, to: &vector)
        append(trunks.quality, to: &vector)
        append(trunks.spike, to: &vector)
    }

    private static func append(_ values: [Double], to buffer: ManasMLXRuntimeFloatTensorBuffer) {
        buffer.append(contentsOf: values)
    }

    private static func append(_ values: [Double], to vector: inout [Float]) {
        for value in values {
            vector.append(Float(value))
        }
    }

    private static func appendGoalVector(
        goals: [ControlGoal],
        size: Int,
        to buffer: ManasMLXRuntimeFloatTensorBuffer
    ) {
        guard size > 0 else { return }
        guard let primary = goals.first else {
            buffer.appendZeros(count: size)
            return
        }
        let copiedCount = min(primary.vector.count, size)
        for index in 0..<copiedCount {
            buffer.append(primary.vector[index])
        }
        buffer.appendZeros(count: size - copiedCount)
    }
}
