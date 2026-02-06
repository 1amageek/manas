import Foundation

public protocol NerveBundle {
    mutating func process(samples: [SignalSample], time: TimeInterval) throws -> NerveBundleOutput
}
