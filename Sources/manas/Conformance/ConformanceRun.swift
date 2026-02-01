public struct ConformanceRun: Sendable, Equatable {
    public let inputs: [InputFrame]
    public let outputs: [[DriveIntent]]

    public init(inputs: [InputFrame], outputs: [[DriveIntent]]) {
        self.inputs = inputs
        self.outputs = outputs
    }
}

