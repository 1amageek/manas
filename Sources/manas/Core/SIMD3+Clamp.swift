public extension SIMD3 where Scalar == Double {
    func clamped(maxMagnitude: Double) -> SIMD3<Double> {
        let magnitude = (x * x + y * y + z * z).squareRoot()
        guard magnitude > maxMagnitude, magnitude > 0 else {
            return self
        }
        let scale = maxMagnitude / magnitude
        return SIMD3<Double>(x * scale, y * scale, z * scale)
    }
}

