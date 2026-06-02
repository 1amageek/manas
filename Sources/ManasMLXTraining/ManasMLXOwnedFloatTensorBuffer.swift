import MLX

public final class ManasMLXOwnedFloatTensorBuffer {
    public let capacity: Int
    public private(set) var count = 0

    private let pointer: UnsafeMutablePointer<Float>
    private let allocatedCapacity: Int
    private var transferred = false

    public init(capacity: Int) {
        precondition(capacity >= 0)
        self.capacity = capacity
        self.allocatedCapacity = max(1, capacity)
        self.pointer = UnsafeMutablePointer<Float>.allocate(capacity: allocatedCapacity)
    }

    deinit {
        guard !transferred else { return }
        pointer.deinitialize(count: count)
        pointer.deallocate()
    }

    public func append(_ value: Float) {
        precondition(count < capacity)
        pointer.advanced(by: count).initialize(to: value)
        count += 1
    }

    public func append(contentsOf values: [Float]) {
        appendContiguousValues(values)
    }

    public func append(contentsOf values: ArraySlice<Float>) {
        appendContiguousValues(values)
    }

    public func append<Values: Collection>(converting values: Values) where Values.Element == Double {
        precondition(count + values.count <= capacity)
        for value in values {
            append(Float(value))
        }
    }

    public func appendZeros(count zeroCount: Int) {
        guard zeroCount > 0 else { return }
        precondition(count + zeroCount <= capacity)
        pointer.advanced(by: count).initialize(repeating: 0, count: zeroCount)
        count += zeroCount
    }

    public func makeArray(shape: [Int]) -> MLXArray {
        precondition(count == capacity)
        transferred = true
        let pointer = self.pointer
        let capacity = self.capacity
        return MLXArray(rawPointer: UnsafeMutableRawPointer(pointer), shape, dtype: .float32) {
            pointer.deinitialize(count: capacity)
            pointer.deallocate()
        }
    }

    public static func makeArray(values: [Float], shape: [Int]) -> MLXArray {
        let buffer = ManasMLXOwnedFloatTensorBuffer(capacity: values.count)
        buffer.append(contentsOf: values)
        return buffer.makeArray(shape: shape)
    }

    public static func makeArray<Values: Collection>(
        converting values: Values,
        shape: [Int]
    ) -> MLXArray where Values.Element == Double {
        let buffer = ManasMLXOwnedFloatTensorBuffer(capacity: values.count)
        buffer.append(converting: values)
        return buffer.makeArray(shape: shape)
    }

    private func appendContiguousValues<Values: Collection>(_ values: Values) where Values.Element == Float {
        guard !values.isEmpty else { return }
        precondition(count + values.count <= capacity)
        if let copied: Void = values.withContiguousStorageIfAvailable({ source in
            guard let sourceBase = source.baseAddress else { return }
            pointer.advanced(by: count).initialize(from: sourceBase, count: source.count)
        }) {
            _ = copied
            count += values.count
        } else {
            appendElementWise(values)
        }
    }

    private func appendElementWise<Values: Collection>(_ values: Values) where Values.Element == Float {
        for value in values {
            append(value)
        }
    }
}
