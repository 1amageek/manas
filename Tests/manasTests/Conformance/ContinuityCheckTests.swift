import Testing
@testable import manas

@Test func continuityCheckPassesSimpleCase() async throws {
    let input = [0.0, 0.0]
    let inputPrime = [0.1, 0.0]
    let output = [0.0, 0.0]
    let outputPrime = [0.05, 0.0]

    let result = ContinuityCheck.evaluate(
        input: input,
        inputPrime: inputPrime,
        output: output,
        outputPrime: outputPrime,
        l2: 1.0,
        lInf: 1.0
    )
    #expect(result.passes == true)
}
