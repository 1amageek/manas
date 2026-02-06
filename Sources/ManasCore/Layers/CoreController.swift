import Foundation

public protocol CoreController {
    mutating func update(trunks: TrunkBundle, time: TimeInterval) throws -> [DriveIntent]
}
