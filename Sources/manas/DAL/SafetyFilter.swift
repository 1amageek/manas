import Foundation

public struct SafetyFilter: Sendable {
    public enum ValidationError: Error, Equatable {
        case missingLimit(ActuatorIndex)
        case nonFiniteDelta
        case nonPositiveDelta
    }

    private let limits: ActuatorLimits
    private var lastCommands: [ActuatorIndex: Double]

    public init(limits: ActuatorLimits) {
        self.limits = limits
        self.lastCommands = [:]
    }

    public mutating func reset() {
        lastCommands = [:]
    }

    public mutating func apply(
        commands: [ActuatorCommand],
        deltaTime: TimeInterval
    ) throws -> [ActuatorCommand] {
        guard deltaTime.isFinite else { throw ValidationError.nonFiniteDelta }
        guard deltaTime > 0 else { throw ValidationError.nonPositiveDelta }

        var filtered: [ActuatorCommand] = []
        filtered.reserveCapacity(commands.count)

        for command in commands {
            guard let limit = limits.limit(for: command.index) else {
                throw ValidationError.missingLimit(command.index)
            }

            let clampedValue = min(max(command.value, limit.range.lowerBound), limit.range.upperBound)
            let finalValue = applyRateLimit(
                index: command.index,
                value: clampedValue,
                maxRate: limit.maxRate,
                deltaTime: deltaTime
            )

            lastCommands[command.index] = finalValue
            filtered.append(try ActuatorCommand(index: command.index, value: finalValue))
        }

        return filtered
    }

    private func applyRateLimit(
        index: ActuatorIndex,
        value: Double,
        maxRate: Double?,
        deltaTime: TimeInterval
    ) -> Double {
        guard let maxRate else {
            return value
        }

        let previous = lastCommands[index] ?? value
        let maxDelta = maxRate * deltaTime
        let delta = value - previous
        let limitedDelta = min(max(delta, -maxDelta), maxDelta)
        return previous + limitedDelta
    }
}

