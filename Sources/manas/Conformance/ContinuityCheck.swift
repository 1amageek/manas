public struct ContinuityCheck: Sendable {
    public struct Result: Sendable, Equatable {
        public let l2Delta: Double
        public let lInfDelta: Double
        public let l2Limit: Double
        public let lInfLimit: Double
        public let passes: Bool
    }

    public static func evaluate(
        input: [Double],
        inputPrime: [Double],
        output: [Double],
        outputPrime: [Double],
        l2: Double,
        lInf: Double
    ) -> Result {
        let inputDelta = zip(input, inputPrime).map { $0 - $1 }
        let outputDelta = zip(output, outputPrime).map { $0 - $1 }

        let l2Delta = ConformanceMath.l2Norm(outputDelta)
        let lInfDelta = ConformanceMath.lInfNorm(outputDelta)

        let l2Limit = l2 * ConformanceMath.l2Norm(inputDelta)
        let lInfLimit = lInf * ConformanceMath.lInfNorm(inputDelta)

        let passes = l2Delta <= l2Limit && lInfDelta <= lInfLimit
        return Result(
            l2Delta: l2Delta,
            lInfDelta: lInfDelta,
            l2Limit: l2Limit,
            lInfLimit: lInfLimit,
            passes: passes
        )
    }
}
