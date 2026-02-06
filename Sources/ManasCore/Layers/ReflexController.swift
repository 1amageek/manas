import Foundation

public protocol ReflexController {
    mutating func update(
        bundle: NerveBundleOutput,
        trunks: TrunkBundle,
        time: TimeInterval
    ) throws -> [ReflexCorrection]
}
