import Configuration

public struct ManasConfigLoader {
    public init() {}

    public func load(reader: ConfigReader) -> ManasRuntimeConfig {
        let levelString = reader.string(forKey: "MANAS_LOG_LEVEL", default: "info")
        let label = reader.string(forKey: "MANAS_LOG_LABEL", default: "manas")
        let level = ManasRuntimeConfig.parseLogLevel(levelString)
        return ManasRuntimeConfig(logLevel: level, logLabel: label)
    }

    public func loadFromEnvironment() -> ManasRuntimeConfig {
        let reader = ConfigReader(provider: EnvironmentVariablesProvider())
        return load(reader: reader)
    }
}
