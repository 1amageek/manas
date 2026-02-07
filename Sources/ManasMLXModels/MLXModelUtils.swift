import MLX

public enum MLXModelUtils {
    /// Normalize input to 3D sequence format `[batch, seq, features]`.
    /// - 1D `[features]` → `[1, 1, features]`
    /// - 2D `[seq, features]` → `[1, seq, features]`
    /// - 3D+ passed through unchanged
    public static func normalizeSequence(_ input: MLXArray) -> MLXArray {
        switch input.ndim {
        case 1:
            return input.reshaped([1, 1, input.shape[0]])
        case 2:
            return input.reshaped([1, input.shape[0], input.shape[1]])
        default:
            return input
        }
    }

    /// Validate auxiliary output: check size and finiteness.
    /// Returns nil if shape mismatch or any non-finite values detected.
    public static func sanitizeAux(_ aux: MLXArray, expectedSize: Int) -> MLXArray? {
        guard aux.shape.last == expectedSize else { return nil }
        let invalid = logicalOr(isNaN(aux), isInf(aux))
        if any(invalid).item(Bool.self) {
            return nil
        }
        return aux
    }
}
