import Foundation

public struct ConfigHash {
    public enum ValidationError: Error, Equatable {
        case encodingFailed
    }

    public static func hash<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(value) else {
            throw ValidationError.encodingFailed
        }
        let digest = FNV1a64.hash(data: [UInt8](data))
        return String(format: "%016llx", digest)
    }
}
