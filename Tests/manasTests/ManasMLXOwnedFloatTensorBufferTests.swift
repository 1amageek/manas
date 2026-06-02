import MLX
import Testing
@testable import ManasMLXTraining

@Test func ownedFloatTensorBufferTransfersValuesIntoMLXArray() {
    let buffer = ManasMLXOwnedFloatTensorBuffer(capacity: 6)
    buffer.append(1)
    buffer.append(contentsOf: [2, 3])
    buffer.append(contentsOf: [4, 5, 6][0..<3])

    let array = buffer.makeArray(shape: [2, 3])

    #expect(array.shape == [2, 3])
    #expect(array.asArray(Float.self) == [1, 2, 3, 4, 5, 6])
}

@Test func ownedFloatTensorBufferConvertsDoubleValuesIntoFloatTensor() {
    let array = ManasMLXOwnedFloatTensorBuffer.makeArray(
        converting: [1.25, -2.5, 3.75],
        shape: [3]
    )

    #expect(array.shape == [3])
    #expect(array.asArray(Float.self) == [1.25, -2.5, 3.75])
}
