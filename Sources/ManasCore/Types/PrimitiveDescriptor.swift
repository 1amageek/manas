import Foundation

public struct PrimitiveDescriptor: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case emptyRange
        case nonFiniteRange
        case negativeParameterCount
    }

    public let id: DriveIndex
    public let name: String?
    public let activationRange: ClosedRange<Double>
    public let parameterCount: Int
    public let semantics: DriveSemantics?

    public init(
        id: DriveIndex,
        name: String? = nil,
        activationRange: ClosedRange<Double> = -1.0...1.0,
        parameterCount: Int = 0,
        semantics: DriveSemantics? = nil
    ) throws {
        guard activationRange.lowerBound.isFinite, activationRange.upperBound.isFinite else {
            throw ValidationError.nonFiniteRange
        }
        guard activationRange.lowerBound <= activationRange.upperBound else {
            throw ValidationError.emptyRange
        }
        guard parameterCount >= 0 else { throw ValidationError.negativeParameterCount }
        self.id = id
        self.name = name
        self.activationRange = activationRange
        self.parameterCount = parameterCount
        self.semantics = semantics
    }
}
