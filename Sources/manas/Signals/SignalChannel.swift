public struct SignalChannel<Index: OrderedIndex> {
    public let index: Index
    public var generator: AnySignalGenerator

    public init(index: Index, generator: AnySignalGenerator) {
        self.index = index
        self.generator = generator
    }
}
