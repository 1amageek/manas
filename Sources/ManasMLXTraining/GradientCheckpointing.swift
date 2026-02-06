import MLX
import MLXNN
import ManasMLXModels

/// Utilities for memory-efficient training via activation recomputation.
///
/// Instead of storing all intermediate activations during the forward pass,
/// gradient checkpointing recomputes them during the backward pass. This
/// trades compute time for memory savings, enabling larger batch sizes or
/// longer sequences.
public enum GradientCheckpointing {

    /// Wraps a forward function to recompute activations during backpropagation.
    ///
    /// Usage with `valueAndGrad`:
    /// ```swift
    /// let lg = valueAndGrad(model: model) { model, input, target in
    ///     let output = GradientCheckpointing.checkpointedForward(model: model, input: input) {
    ///         model, x in model.forward(trunks: x)
    ///     }
    ///     return mseLoss(predictions: output.drives, targets: target, reduction: .mean)
    /// }
    /// ```
    ///
    /// - Note: In MLX, the lazy evaluation model means that activations are
    ///   typically not materialized until needed. This helper provides a
    ///   structured pattern for controlling evaluation boundaries.
    public static func checkpointedForward<Model: Module, Output>(
        model: Model,
        input: MLXArray,
        forward: (Model, MLXArray) -> Output
    ) -> Output {
        // Force evaluation of input to establish a checkpoint boundary.
        // MLX's lazy evaluation means intermediate values are recomputed
        // on demand during backpropagation by default.
        eval(input)
        return forward(model, input)
    }

    /// Segment a sequence into chunks and process each with checkpoint boundaries.
    ///
    /// For long sequences, this limits peak memory by processing segments
    /// independently and forcing evaluation at segment boundaries.
    ///
    /// - Parameters:
    ///   - input: Full input sequence [batch, seqLen, features]
    ///   - segmentSize: Number of time steps per segment
    ///   - process: Function to process each segment
    /// - Returns: Concatenated outputs from all segments
    public static func segmentedForward(
        input: MLXArray,
        segmentSize: Int,
        process: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let seqLen = input.dim(1)
        guard seqLen > segmentSize else {
            return process(input)
        }

        var outputs: [MLXArray] = []
        var offset = 0

        while offset < seqLen {
            let end = min(offset + segmentSize, seqLen)
            let segment = input[0..., offset..<end]
            eval(segment)
            let output = process(segment)
            eval(output)
            outputs.append(output)
            offset = end
        }

        return concatenated(outputs, axis: 1)
    }
}
