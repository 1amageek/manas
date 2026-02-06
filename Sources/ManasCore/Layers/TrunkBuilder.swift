import Foundation

public protocol TrunkBuilder {
    mutating func build(from gated: GatedBundle, time: TimeInterval) throws -> TrunkBundle
}
