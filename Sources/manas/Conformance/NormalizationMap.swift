public struct NormalizationMap<Index: OrderedIndex>: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case nonFinite(Index)
        case nonPositive(Index)
        case missingValue(Index)
        case unexpectedValue(Index)
    }

    private let ranges: [Index: Double]

    public init(ranges: [Index: Double]) throws {
        for (index, range) in ranges {
            guard range.isFinite else { throw ValidationError.nonFinite(index) }
            guard range > 0 else { throw ValidationError.nonPositive(index) }
        }
        self.ranges = ranges
    }

    public func normalizedValue(for index: Index, value: Double) throws -> Double {
        guard let range = ranges[index] else {
            throw ValidationError.missingValue(index)
        }
        return value / range
    }

    public func range(for index: Index) throws -> Double {
        guard let range = ranges[index] else {
            throw ValidationError.missingValue(index)
        }
        return range
    }

    public func normalizedVector(values: [Index: Double]) throws -> [Double] {
        var result: [Double] = []
        result.reserveCapacity(ranges.count)

        for index in ranges.keys.sorted() {
            guard let value = values[index] else {
                throw ValidationError.missingValue(index)
            }
            result.append(value / ranges[index]!)
        }

        for index in values.keys where ranges[index] == nil {
            throw ValidationError.unexpectedValue(index)
        }

        return result
    }

    public var indices: [Index] {
        ranges.keys.sorted()
    }
}
