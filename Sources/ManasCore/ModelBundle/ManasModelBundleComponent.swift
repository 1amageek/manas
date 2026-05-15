public struct ManasModelBundleComponent: Codable, Sendable, Equatable {
    public let role: ManasModelBundleComponentRole
    public let path: String
    public let contentType: String
    public let required: Bool
    public let byteCount: Int?
    public let fnv1a64: String?

    public init(
        role: ManasModelBundleComponentRole,
        path: String,
        contentType: String,
        required: Bool = true,
        byteCount: Int? = nil,
        fnv1a64: String? = nil
    ) {
        self.role = role
        self.path = path
        self.contentType = contentType
        self.required = required
        self.byteCount = byteCount
        self.fnv1a64 = fnv1a64
    }
}
