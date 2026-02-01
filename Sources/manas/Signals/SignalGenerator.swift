import Foundation

public protocol SignalGenerator {
    mutating func sample(at time: TimeInterval) throws -> Double
}
