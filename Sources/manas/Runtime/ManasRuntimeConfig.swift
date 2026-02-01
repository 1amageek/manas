import Logging

public struct ManasRuntimeConfig: Sendable, Equatable {
    public let logLevel: Logger.Level
    public let logLabel: String

    public init(logLevel: Logger.Level, logLabel: String) {
        self.logLevel = logLevel
        self.logLabel = logLabel
    }

    public static let `default` = ManasRuntimeConfig(logLevel: .info, logLabel: "manas")

    public static func parseLogLevel(_ value: String) -> Logger.Level {
        switch value.lowercased() {
        case "trace": return .trace
        case "debug": return .debug
        case "info": return .info
        case "notice": return .notice
        case "warning": return .warning
        case "error": return .error
        case "critical": return .critical
        default: return .info
        }
    }
}
