public struct IMU6Mapping: Sendable, Codable, Equatable {
    public let gyroX: PhaseIndex
    public let gyroY: PhaseIndex
    public let gyroZ: PhaseIndex
    public let accelX: PhaseIndex
    public let accelY: PhaseIndex
    public let accelZ: PhaseIndex

    public init(
        gyroX: PhaseIndex = PhaseIndex(0),
        gyroY: PhaseIndex = PhaseIndex(1),
        gyroZ: PhaseIndex = PhaseIndex(2),
        accelX: PhaseIndex = PhaseIndex(3),
        accelY: PhaseIndex = PhaseIndex(4),
        accelZ: PhaseIndex = PhaseIndex(5)
    ) {
        self.gyroX = gyroX
        self.gyroY = gyroY
        self.gyroZ = gyroZ
        self.accelX = accelX
        self.accelY = accelY
        self.accelZ = accelZ
    }
}

