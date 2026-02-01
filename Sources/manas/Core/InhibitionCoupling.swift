public struct InhibitionCoupling: Sendable, Codable, Equatable {
    public enum ValidationError: Error, Equatable {
        case outOfRange(PerceptionIndex, DriveIndex, Double)
    }

    private let matrix: [PerceptionIndex: [DriveIndex: Double]]

    public init(matrix: [PerceptionIndex: [DriveIndex: Double]]) throws {
        for (perception, row) in matrix {
            for (drive, value) in row {
                guard value.isFinite, value >= 0, value <= 1 else {
                    throw ValidationError.outOfRange(perception, drive, value)
                }
            }
        }

        self.matrix = matrix
    }

    public func coupling(for perception: PerceptionIndex, drive: DriveIndex) -> Double {
        matrix[perception]?[drive] ?? 0.0
    }
}

