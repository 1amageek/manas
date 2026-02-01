import Foundation

public struct AnySignalGenerator: SignalGenerator {
    private var box: AnySignalGeneratorBox

    public init<Generator: SignalGenerator>(_ generator: Generator) {
        self.box = ConcreteSignalGeneratorBox(generator)
    }

    public mutating func sample(at time: TimeInterval) throws -> Double {
        try box.sample(at: time)
    }
}

private protocol AnySignalGeneratorBox {
    mutating func sample(at time: TimeInterval) throws -> Double
}

private struct ConcreteSignalGeneratorBox<Generator: SignalGenerator>: AnySignalGeneratorBox {
    private var generator: Generator

    init(_ generator: Generator) {
        self.generator = generator
    }

    mutating func sample(at time: TimeInterval) throws -> Double {
        try generator.sample(at: time)
    }
}

