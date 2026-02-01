public struct ConformanceInputPair {
    public var base: InputFamily
    public var perturbed: InputFamily

    public init(base: InputFamily, perturbed: InputFamily) {
        self.base = base
        self.perturbed = perturbed
    }
}
