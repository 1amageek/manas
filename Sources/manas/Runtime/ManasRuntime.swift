import Configuration
import Logging

public struct ManasRuntime {
    public let config: ManasRuntimeConfig
    public let logger: Logger

    public init(
        loader: ManasConfigLoader = ManasConfigLoader(),
        reader: ConfigReader? = nil
    ) {
        let config: ManasRuntimeConfig
        if let reader {
            config = loader.load(reader: reader)
        } else {
            config = ManasRuntimeConfig.default
        }
        let loggerFactory = ManasLoggerFactory()
        self.config = config
        self.logger = loggerFactory.make(label: config.logLabel, level: config.logLevel)
    }

    public static func fromEnvironment() -> ManasRuntime {
        let reader = ConfigReader(provider: EnvironmentVariablesProvider())
        return ManasRuntime(reader: reader)
    }
}
