import Foundation

public protocol Gating {
    mutating func apply(bundle: NerveBundleOutput, time: TimeInterval) throws -> GatedBundle
}
