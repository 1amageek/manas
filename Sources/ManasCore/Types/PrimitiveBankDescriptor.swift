import Foundation

public struct PrimitiveBankDescriptor: Hashable, Sendable, Codable {
    public enum ValidationError: Error, Equatable {
        case nonFiniteRange
        case emptyRange
        case invalidDriveCount
        case duplicatePrimitive(DriveIndex)
        case missingPrimitive(DriveIndex)
        case outOfRangePrimitive(DriveIndex)
    }

    public let driveCount: Int
    public let primitives: [PrimitiveDescriptor]
    public let defaultActivationRange: ClosedRange<Double>

    public init(
        driveCount: Int,
        primitives: [PrimitiveDescriptor]? = nil,
        defaultActivationRange: ClosedRange<Double> = -1.0...1.0
    ) throws {
        guard driveCount >= 0 else { throw ValidationError.invalidDriveCount }
        guard defaultActivationRange.lowerBound.isFinite,
              defaultActivationRange.upperBound.isFinite else {
            throw ValidationError.nonFiniteRange
        }
        guard defaultActivationRange.lowerBound <= defaultActivationRange.upperBound else {
            throw ValidationError.emptyRange
        }

        if let primitives {
            guard primitives.count == driveCount else {
                throw ValidationError.invalidDriveCount
            }
            var seen: Set<DriveIndex> = []
            for primitive in primitives {
                guard primitive.id.rawValue < UInt32(driveCount) else {
                    throw ValidationError.outOfRangePrimitive(primitive.id)
                }
                guard seen.insert(primitive.id).inserted else {
                    throw ValidationError.duplicatePrimitive(primitive.id)
                }
            }
            for index in 0..<driveCount {
                let id = DriveIndex(UInt32(index))
                guard seen.contains(id) else {
                    throw ValidationError.missingPrimitive(id)
                }
            }
            self.primitives = primitives
        } else {
            var generated: [PrimitiveDescriptor] = []
            generated.reserveCapacity(driveCount)
            for index in 0..<driveCount {
                let id = DriveIndex(UInt32(index))
                let primitive = try PrimitiveDescriptor(
                    id: id,
                    name: nil,
                    activationRange: defaultActivationRange,
                    parameterCount: 0
                )
                generated.append(primitive)
            }
            self.primitives = generated
        }
        self.driveCount = driveCount
        self.defaultActivationRange = defaultActivationRange
    }

    public func descriptor(for id: DriveIndex) throws -> PrimitiveDescriptor {
        guard id.rawValue < UInt32(driveCount) else {
            throw ValidationError.outOfRangePrimitive(id)
        }
        guard let descriptor = primitives.first(where: { $0.id == id }) else {
            throw ValidationError.missingPrimitive(id)
        }
        return descriptor
    }

    public func activationRange(for id: DriveIndex) throws -> ClosedRange<Double> {
        try descriptor(for: id).activationRange
    }
}
