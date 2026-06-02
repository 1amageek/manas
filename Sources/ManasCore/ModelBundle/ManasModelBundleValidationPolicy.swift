public struct ManasModelBundleValidationPolicy: Sendable, Equatable {
    public let validateComponentByteCounts: Bool
    public let validateComponentDigests: Bool

    public init(
        validateComponentByteCounts: Bool = true,
        validateComponentDigests: Bool = true
    ) {
        self.validateComponentByteCounts = validateComponentByteCounts
        self.validateComponentDigests = validateComponentDigests
    }

    public static let full = ManasModelBundleValidationPolicy()

    public static let componentMetadata = ManasModelBundleValidationPolicy(
        validateComponentByteCounts: true,
        validateComponentDigests: false
    )
}
