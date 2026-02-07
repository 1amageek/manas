import MLX
import MLXRandom
import MLXNN

public final class ManasMLXCore: Module {
    public let config: ManasMLXCoreConfig

    @ModuleInfo public var encoder1: Linear
    @ModuleInfo public var encoder2: Linear
    @ModuleInfo public var fastGRU: GRU
    @ModuleInfo public var slowGRU: GRU
    @ModuleInfo public var driveHead: Linear
    @ModuleInfo public var auxHead: Linear

    // Type embeddings for channel-agnostic tokenization
    @ModuleInfo public var typeEmbeddings: ChannelTypeEmbeddingTable?

    // Shared encoder/decoder for nerve network mode
    @ModuleInfo public var sharedEncoder: SharedEncoder?
    @ModuleInfo public var sharedActuatorDecoder: SharedActuatorDecoder?
    @ModuleInfo public var morphologyEncoder: MorphologyEncoder?

    // Adaptation module (1D CNN → environment parameter vector z)
    @ModuleInfo public var adaptationModule: AdaptationModule?

    // Descending channels (GoalCore pattern: encode → gate → concat before GRU)
    @ModuleInfo public var descendingEncoder: Linear?
    @ModuleInfo public var descendingGate: Linear?

    // RSSM posterior: q(z_t | h_t, x_t)
    @ModuleInfo public var posteriorNet: Linear?
    // RSSM prior: p(z_t | h_t)
    @ModuleInfo public var priorNet: Linear?
    // Imagination transition input: concat(z_{t-1}, drives_{t-1}) → embeddingSize
    @ModuleInfo public var transitionInputProj: Linear?

    // Prediction heads (2-layer MLPs)
    @ModuleInfo public var rewardHead1: Linear?
    @ModuleInfo public var rewardHead2: Linear?
    @ModuleInfo public var continueHead1: Linear?
    @ModuleInfo public var continueHead2: Linear?
    @ModuleInfo public var valueHead1: Linear?
    @ModuleInfo public var valueHead2: Linear?

    public init(config: ManasMLXCoreConfig) {
        self.config = config
        if config.sharedEncoderEnabled {
            let sharedInputSize = 1 + config.typeEmbeddingDim
            self._sharedEncoder.wrappedValue = SharedEncoder(inputSize: sharedInputSize, embeddingSize: config.embeddingSize)
            self._encoder1.wrappedValue = Linear(sharedInputSize, config.embeddingSize)  // placeholder, unused in shared path
            self._typeEmbeddings.wrappedValue = ChannelTypeEmbeddingTable(
                channelCount: config.typeEmbeddingCount,
                embeddingDim: config.typeEmbeddingDim
            )
            if config.morphologyDim > 0 {
                self._morphologyEncoder.wrappedValue = MorphologyEncoder(
                    morphologyDim: config.morphologyDim,
                    embeddingSize: config.embeddingSize
                )
            }
        } else if config.typeEmbeddingEnabled {
            self._encoder1.wrappedValue = Linear(1 + config.typeEmbeddingDim, config.embeddingSize)
            self._typeEmbeddings.wrappedValue = ChannelTypeEmbeddingTable(
                channelCount: config.typeEmbeddingCount,
                embeddingDim: config.typeEmbeddingDim
            )
        } else {
            self._encoder1.wrappedValue = Linear(config.inputSize, config.embeddingSize)
        }
        self._encoder2.wrappedValue = Linear(config.embeddingSize, config.embeddingSize)

        if config.descendingEnabled && !config.sharedEncoderEnabled {
            self._descendingEncoder.wrappedValue = Linear(config.descendingSize, config.descendingEmbeddingSize)
            self._descendingGate.wrappedValue = Linear(config.descendingEmbeddingSize, config.descendingEmbeddingSize)
        }

        if config.adaptationEnabled {
            self._adaptationModule.wrappedValue = AdaptationModule(
                historyLength: config.adaptationHistoryLength,
                inputDim: config.embeddingSize,
                outputDim: config.adaptationOutputDim
            )
        }

        self._fastGRU.wrappedValue = GRU(inputSize: config.gruInputSize, hiddenSize: config.fastHiddenSize)
        self._slowGRU.wrappedValue = GRU(inputSize: config.gruInputSize, hiddenSize: config.slowHiddenSize)

        let effectiveDriveCount = config.actuatorCount > 0 ? config.actuatorCount : config.driveCount

        let featureSize = config.fullFeatureSize

        if config.sharedDecoderEnabled {
            let decoderInputSize = featureSize + config.typeEmbeddingDim
            self._sharedActuatorDecoder.wrappedValue = SharedActuatorDecoder(
                inputSize: decoderInputSize,
                hiddenSize: config.decoderHiddenSize
            )
            self._driveHead.wrappedValue = Linear(featureSize, effectiveDriveCount)
            self._auxHead.wrappedValue = Linear(featureSize, max(config.auxSize, 1))
        }

        if config.rssmEnabled {
            self._posteriorNet.wrappedValue = Linear(config.deterministicSize + config.embeddingSize, config.stochasticLatentSize)
            self._priorNet.wrappedValue = Linear(config.deterministicSize, config.stochasticLatentSize)
            self._transitionInputProj.wrappedValue = Linear(config.stochasticLatentSize + effectiveDriveCount, config.embeddingSize)
            if !config.sharedDecoderEnabled {
                self._driveHead.wrappedValue = Linear(featureSize, config.driveCount)
                self._auxHead.wrappedValue = Linear(featureSize, max(config.auxSize, 1))
            }

            if config.rewardHeadHiddenSize > 0 {
                self._rewardHead1.wrappedValue = Linear(featureSize, config.rewardHeadHiddenSize)
                self._rewardHead2.wrappedValue = Linear(config.rewardHeadHiddenSize, 1)
            }
            if config.continueHeadHiddenSize > 0 {
                self._continueHead1.wrappedValue = Linear(featureSize, config.continueHeadHiddenSize)
                self._continueHead2.wrappedValue = Linear(config.continueHeadHiddenSize, 1)
            }
            if config.valueHeadHiddenSize > 0 {
                self._valueHead1.wrappedValue = Linear(featureSize, config.valueHeadHiddenSize)
                self._valueHead2.wrappedValue = Linear(config.valueHeadHiddenSize, 1)
            }
        } else if !config.sharedDecoderEnabled {
            self._driveHead.wrappedValue = Linear(featureSize, config.driveCount)
            self._auxHead.wrappedValue = Linear(featureSize, config.auxSize)
        }
    }

    // MARK: - Forward (backward compatible)

    public func forward(
        trunks: MLXArray,
        descending: MLXArray? = nil,
        morphology: MLXArray? = nil,
        state: ManasMLXCoreState? = nil
    ) -> ManasMLXCoreOutput {
        let (gruInput, embedded) = encodeInput(
            trunks: trunks, descending: descending, morphology: morphology
        )

        let fastSeq = fastGRU(gruInput, hidden: state?.fast)
        let slowSeq = slowGRU(gruInput, hidden: state?.slow)
        let h_t = concatenated([fastSeq, slowSeq], axis: -1)

        let combined: MLXArray
        let lastZ: MLXArray?
        if config.rssmEnabled, let posteriorNet {
            let posteriorInput = concatenated([h_t, embedded], axis: -1)
            let posteriorLogits = posteriorNet(posteriorInput)
            let z_t = sampleCategorical(logits: posteriorLogits)
            combined = concatenated([h_t, z_t], axis: -1)
            let lastIndex = z_t.dim(-2) - 1
            lastZ = z_t[.ellipsis, lastIndex, 0...]
        } else {
            combined = h_t
            lastZ = nil
        }

        let (features, nextHistory, adaptZ) = applyAdaptation(
            combined: combined, embedded: embedded, state: state
        )
        let drives = decodeDrives(features)

        let auxOutput: MLXArray?
        if config.auxEnabled {
            auxOutput = sanitizeAux(auxHead(features))
        } else {
            auxOutput = nil
        }

        let lastIndex = h_t.dim(-2) - 1
        let nextFast = fastSeq[.ellipsis, lastIndex, 0...]
        let nextSlow = slowSeq[.ellipsis, lastIndex, 0...]

        return ManasMLXCoreOutput(
            drives: drives,
            nextState: ManasMLXCoreState(
                fast: nextFast, slow: nextSlow, z: lastZ,
                ascendingHistory: nextHistory, adaptationZ: adaptZ
            ),
            aux: auxOutput
        )
    }

    public func callAsFunction(_ trunks: MLXArray) -> MLXArray {
        forward(trunks: trunks, state: nil).drives
    }

    // MARK: - RSSM Forward (world model training)

    public func forwardRSSM(
        trunks: MLXArray,
        descending: MLXArray? = nil,
        morphology: MLXArray? = nil,
        state: ManasMLXCoreState? = nil
    ) -> RSSMOutput {
        let (gruInput, embedded) = encodeInput(
            trunks: trunks, descending: descending, morphology: morphology
        )

        let fastSeq = fastGRU(gruInput, hidden: state?.fast)
        let slowSeq = slowGRU(gruInput, hidden: state?.slow)
        let h_t = concatenated([fastSeq, slowSeq], axis: -1)

        let posteriorInput = concatenated([h_t, embedded], axis: -1)
        let posteriorLogits = posteriorNet!(posteriorInput)
        let posteriorZ = sampleCategorical(logits: posteriorLogits)

        let priorLogits = priorNet!(h_t)

        let combinedState = concatenated([h_t, posteriorZ], axis: -1)

        let (features, _, adaptZ) = applyAdaptation(
            combined: combinedState, embedded: embedded, state: state
        )

        let drives = decodeDrives(features)

        let trunkPred: MLXArray?
        if config.trunkPredictorEnabled {
            trunkPred = auxHead(features)
        } else {
            trunkPred = nil
        }

        let rewardPred = predictReward(features)
        let continuePred = predictContinue(features)

        let lastIndex = h_t.dim(-2) - 1
        let nextFast = fastSeq[.ellipsis, lastIndex, 0...]
        let nextSlow = slowSeq[.ellipsis, lastIndex, 0...]
        let nextZ = posteriorZ[.ellipsis, lastIndex, 0...]

        return RSSMOutput(
            drives: drives,
            posteriorLogits: posteriorLogits,
            priorLogits: priorLogits,
            posteriorZ: posteriorZ,
            trunkPrediction: trunkPred,
            rewardPrediction: rewardPred,
            continuePrediction: continuePred,
            nextState: ManasMLXCoreState(
                fast: nextFast, slow: nextSlow, z: nextZ,
                adaptationZ: adaptZ
            )
        )
    }

    // MARK: - Imagination Step (prior only, no observation)

    public func imaginationStep(
        action: MLXArray,
        descending: MLXArray? = nil,
        state: ManasMLXCoreState
    ) -> RSSMOutput {
        let z_prev = state.z ?? MLXArray.zeros([action.dim(0), config.stochasticLatentSize])
        let transInput = transitionInputProj!(concatenated([z_prev, action], axis: -1))
        let transInputSeq = expandedDimensions(transInput, axis: 1)

        let fastSeq = fastGRU(transInputSeq, hidden: state.fast)
        let slowSeq = slowGRU(transInputSeq, hidden: state.slow)
        let h_t = concatenated([fastSeq, slowSeq], axis: -1)

        let priorLogits = priorNet!(h_t)
        let z_t = sampleCategorical(logits: priorLogits)

        let combinedState = concatenated([h_t, z_t], axis: -1)

        // Adaptation: use frozen z from last real observation (no new history update)
        let features: MLXArray
        let adaptZ: MLXArray?
        if config.adaptationEnabled {
            if let storedZ = state.adaptationZ {
                let zExp = expandedDimensions(storedZ, axis: 1)
                features = concatenated([combinedState, zExp], axis: -1)
                adaptZ = storedZ
            } else {
                let zeroZ = MLXArray.zeros([combinedState.dim(0), 1, config.adaptationOutputDim])
                features = concatenated([combinedState, zeroZ], axis: -1)
                adaptZ = nil
            }
        } else {
            features = combinedState
            adaptZ = nil
        }

        let drives = decodeDrives(features)

        let rewardPred = predictReward(features)
        let continuePred = predictContinue(features)

        let nextFast = fastSeq[.ellipsis, 0, 0...]
        let nextSlow = slowSeq[.ellipsis, 0, 0...]
        let nextZ = z_t[.ellipsis, 0, 0...]

        return RSSMOutput(
            drives: drives,
            posteriorLogits: priorLogits,
            priorLogits: priorLogits,
            posteriorZ: z_t,
            trunkPrediction: nil,
            rewardPrediction: rewardPred,
            continuePrediction: continuePred,
            nextState: ManasMLXCoreState(
                fast: nextFast, slow: nextSlow, z: nextZ,
                adaptationZ: adaptZ
            )
        )
    }

    // MARK: - Value Prediction

    public func predictValue(_ state: MLXArray) -> MLXArray? {
        guard let vh1 = valueHead1, let vh2 = valueHead2 else { return nil }
        return vh2(tanh(vh1(state)))
    }

    // MARK: - Imagination Helpers (public, used by ImaginationTrainer)

    /// Builds full-size feature vector from imagination state (h + z + adaptZ).
    /// Returns [batch, fullFeatureSize] for predictValue / decodeRawDrives.
    public func imaginationFeatures(from state: ManasMLXCoreState) -> MLXArray {
        let h = concatenated([state.fast, state.slow], axis: -1)
        let z = state.z ?? MLXArray.zeros([state.fast.dim(0), config.stochasticLatentSize])
        var features = concatenated([h, z], axis: -1)
        if config.adaptationEnabled {
            let adaptZ = state.adaptationZ ?? MLXArray.zeros([state.fast.dim(0), config.adaptationOutputDim])
            features = concatenated([features, adaptZ], axis: -1)
        }
        return features
    }

    /// Decodes raw (pre-tanh) drive commands from features.
    /// Used by ImaginationTrainer for entropy proxy computation.
    public func decodeRawDrives(_ features: MLXArray) -> MLXArray {
        if config.sharedDecoderEnabled {
            return decodeActuatorsRaw(features: features)
        } else {
            return driveHead(features)
        }
    }

    // MARK: - Internal Helpers (accessible by NerveLoRACore)

    /// Encodes all input channels into GRU input and embedded representation.
    /// `encoderFn` overrides sharedEncoder (used by NerveLoRACore for LoRA injection).
    func encodeInput(
        trunks: MLXArray,
        descending: MLXArray?,
        morphology: MLXArray?,
        encoderFn: ((MLXArray) -> MLXArray)? = nil
    ) -> (gruInput: MLXArray, embedded: MLXArray) {
        if config.sharedEncoderEnabled, let typeEmbeddings {
            let encode: (MLXArray) -> MLXArray
            if let fn = encoderFn {
                encode = fn
            } else {
                let se = sharedEncoder!
                encode = { x in se(x) }
            }

            let ascPool = encodeChannelsWithFn(
                values: trunks, typeIndices: config.ascendingTypeIndices,
                typeEmbeddings: typeEmbeddings, encoderFn: encode
            )

            var pools: [MLXArray] = [ascPool]
            let targetSeq = ascPool.dim(1)

            if config.descendingEnabled {
                if let descInput = descending, let descIndices = config.descendingTypeIndices {
                    let descPool = encodeChannelsWithFn(
                        values: descInput, typeIndices: descIndices,
                        typeEmbeddings: typeEmbeddings, encoderFn: encode
                    )
                    pools.append(descPool)
                } else {
                    pools.append(MLXArray.zeros([ascPool.dim(0), targetSeq, config.embeddingSize]))
                }
            }

            if config.morphologyDim > 0 {
                if let morphologyEncoder, let morphInput = morphology {
                    let morphToken = morphologyEncoder(morphInput)
                    let morphExp = morphToken.ndim == 2
                        ? expandedDimensions(morphToken, axis: 1) : morphToken
                    pools.append(morphExp)
                } else {
                    pools.append(MLXArray.zeros([ascPool.dim(0), targetSeq, config.embeddingSize]))
                }
            }

            // Broadcast all pools to match targetSeq (fix: morphology/descending may be seq=1)
            let broadcasted = pools.map { pool -> MLXArray in
                if pool.dim(1) == 1 && targetSeq > 1 {
                    return repeated(pool, count: targetSeq, axis: 1)
                }
                return pool
            }

            return (concatenated(broadcasted, axis: -1), ascPool)
        } else {
            let embedded = encodeTrunks(trunks)
            let gruInput = applyDescending(embedded: embedded, descending: descending)
            return (gruInput, embedded)
        }
    }

    /// Decodes features into drive commands using shared decoder or driveHead.
    /// `decoderFn` overrides sharedActuatorDecoder (used by NerveLoRACore for LoRA injection).
    public func decodeDrives(
        _ features: MLXArray,
        decoderFn: ((MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        if config.sharedDecoderEnabled {
            return decodeActuatorsWithFn(features: features, decoderFn: decoderFn)
        } else {
            return tanh(driveHead(features)) * config.driveScale
        }
    }

    /// Applies adaptation module to combined state.
    /// Returns (features, nextHistory, adaptZ) where adaptZ is the raw z for state storage.
    func applyAdaptation(
        combined: MLXArray,
        embedded: MLXArray,
        state: ManasMLXCoreState?
    ) -> (features: MLXArray, nextHistory: MLXArray?, adaptZ: MLXArray?) {
        guard config.adaptationEnabled, let adaptationModule else {
            return (combined, nil, nil)
        }

        let histLen = config.adaptationHistoryLength
        let lastEmbedded = embedded[.ellipsis, embedded.dim(1) - 1, 0...]

        let updatedHistory: MLXArray
        if let prevHistory = state?.ascendingHistory {
            let currentLen = prevHistory.dim(1)
            if currentLen >= histLen {
                let trimmed = prevHistory[.ellipsis, 1..., 0...]
                let newStep = expandedDimensions(lastEmbedded, axis: 1)
                updatedHistory = concatenated([trimmed, newStep], axis: 1)
            } else {
                let newStep = expandedDimensions(lastEmbedded, axis: 1)
                updatedHistory = concatenated([prevHistory, newStep], axis: 1)
            }
        } else {
            updatedHistory = expandedDimensions(lastEmbedded, axis: 1)
        }

        if updatedHistory.dim(1) >= histLen {
            let z = adaptationModule(updatedHistory)
            let zExp = expandedDimensions(z, axis: 1)
            let zBroad = repeated(zExp, count: combined.dim(1), axis: 1)
            let features = concatenated([combined, zBroad], axis: -1)
            return (features, updatedHistory, z)
        } else {
            let zeroZ = MLXArray.zeros([combined.dim(0), combined.dim(1), config.adaptationOutputDim])
            let features = concatenated([combined, zeroZ], axis: -1)
            return (features, updatedHistory, nil)
        }
    }

    /// DreamerV3-style categorical sampling with unimix and straight-through gradient.
    func sampleCategorical(logits: MLXArray) -> MLXArray {
        let cats = config.stochasticCategories
        let cls = config.stochasticClasses
        let shape = logits.shape
        let batchAndSeq = Array(shape.dropLast())
        let reshaped = logits.reshaped(batchAndSeq + [cats, cls])

        var probs = softmax(reshaped, axis: -1)

        let unimix = config.stochasticUnimixRatio
        if unimix > 0 {
            let uniform = MLXArray.ones(like: probs) * (1.0 / Float(cls))
            probs = (1.0 - unimix) * probs + unimix * uniform
        }

        if self.training {
            let u = MLXRandom.uniform(0 ..< 1, probs.shape)
            let gumbel = -log(-log(u + 1e-8) + 1e-8)
            let noisyLogits = log(probs + 1e-8) + gumbel
            let sharpProbs = softmax(noisyLogits * 10.0, axis: -1)
            let stProbs = probs + stopGradient(sharpProbs - probs)
            return stProbs.reshaped(batchAndSeq + [cats * cls])
        } else {
            let hardProbs = softmax(reshaped * 100.0, axis: -1)
            return hardProbs.reshaped(batchAndSeq + [cats * cls])
        }
    }

    // MARK: - Private

    private func predictReward(_ combinedState: MLXArray) -> MLXArray? {
        guard let rh1 = rewardHead1, let rh2 = rewardHead2 else { return nil }
        return rh2(tanh(rh1(combinedState)))
    }

    private func predictContinue(_ combinedState: MLXArray) -> MLXArray? {
        guard let ch1 = continueHead1, let ch2 = continueHead2 else { return nil }
        return ch2(tanh(ch1(combinedState)))
    }

    /// Encodes channels using an encoder function with type embeddings (batched, no loops).
    /// Input values: [batch, seq, N] or [batch, N] → Output: [batch, seq, embeddingSize]
    private func encodeChannelsWithFn(
        values: MLXArray,
        typeIndices: [Int]?,
        typeEmbeddings: ChannelTypeEmbeddingTable,
        encoderFn: (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard let indices = typeIndices else {
            return MLXArray.zeros([values.dim(0), 1, config.embeddingSize])
        }

        let normalized = normalizeSequence(values)
        let batch = normalized.dim(0)
        let seq = normalized.dim(1)
        let N = indices.count

        let flat = normalized.reshaped([batch * seq, N])
        let valuesExp = expandedDimensions(flat, axis: -1)

        let typeIdx = MLXArray(indices.map(Int32.init))
        let typeEmbs = typeEmbeddings(typeIdx)
        let typeEmbsBroad = repeated(expandedDimensions(typeEmbs, axis: 0), count: batch * seq, axis: 0)

        let tokenInputs = concatenated([valuesExp, typeEmbsBroad], axis: -1)
        let tokens = encoderFn(tokenInputs)

        let pooled = mean(tokens, axis: 1)
        return pooled.reshaped([batch, seq, config.embeddingSize])
    }

    /// Returns raw (pre-tanh) actuator commands using shared decoder with type embeddings.
    private func decodeActuatorsRaw(
        features: MLXArray,
        decoderFn: ((MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        guard let typeEmbeddings, let actIndices = config.actuatorTypeIndices else {
            return driveHead(features)
        }

        let decode: (MLXArray) -> MLXArray
        if let fn = decoderFn {
            decode = fn
        } else {
            let sad = sharedActuatorDecoder!
            decode = { x in sad(x) }
        }

        let M = actIndices.count
        let batch = features.dim(0)
        let seq = features.dim(1)

        let featFlat = features.reshaped([batch * seq, features.dim(-1)])
        let featExp = expandedDimensions(featFlat, axis: 1)
        let featBroad = repeated(featExp, count: M, axis: 1)

        let actIdx = MLXArray(actIndices.map(Int32.init))
        let actTypeEmbs = typeEmbeddings(actIdx)
        let actTypeBroad = repeated(expandedDimensions(actTypeEmbs, axis: 0), count: batch * seq, axis: 0)

        let decInput = concatenated([featBroad, actTypeBroad], axis: -1)
        let rawCommands = decode(decInput)

        let commands = rawCommands.squeezed(axis: -1)
        return commands.reshaped([batch, seq, M])
    }

    /// Decodes actuator commands using a decoder function with type embeddings (batched, no loops).
    private func decodeActuatorsWithFn(
        features: MLXArray,
        decoderFn: ((MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        let raw = decodeActuatorsRaw(features: features, decoderFn: decoderFn)
        return tanh(raw) * config.driveScale
    }

    /// Encodes trunks into embedding space (legacy and typeEmbedding-only paths).
    private func encodeTrunks(_ trunks: MLXArray) -> MLXArray {
        let sequence = normalizeSequence(trunks)

        if config.typeEmbeddingEnabled, let typeEmbeddings, let indices = config.ascendingTypeIndices {
            let batch = sequence.dim(0)
            let seq = sequence.dim(1)
            let N = indices.count

            let flat = sequence.reshaped([batch * seq, N])
            let valuesExp = expandedDimensions(flat, axis: -1)
            let typeIndices = MLXArray(indices.map(Int32.init))
            let typeEmbs = typeEmbeddings(typeIndices)
            let typeEmbsExp = expandedDimensions(typeEmbs, axis: 0)
            let typeEmbsBroad = repeated(typeEmbsExp, count: batch * seq, axis: 0)

            let tokenInputs = concatenated([valuesExp, typeEmbsBroad], axis: -1)

            var tokens = tanh(encoder1(tokenInputs))
            tokens = tanh(encoder2(tokens))

            let pooled = mean(tokens, axis: 1)
            return pooled.reshaped([batch, seq, config.embeddingSize])
        } else {
            var embedded = encoder1(sequence)
            embedded = tanh(embedded)
            embedded = encoder2(embedded)
            embedded = tanh(embedded)
            return embedded
        }
    }

    /// Applies descending channel gating to embedded trunks (GoalCore pattern).
    private func applyDescending(embedded: MLXArray, descending: MLXArray?) -> MLXArray {
        guard config.descendingEnabled,
              let descendingEncoder,
              let descendingGate,
              let descInput = descending else {
            return embedded
        }

        let descEmbedded = tanh(descendingEncoder(descInput))
        let gate = sigmoid(descendingGate(descEmbedded))
        let gatedDesc = descEmbedded * gate

        let descBroadcast: MLXArray
        if gatedDesc.ndim == 2 {
            let expanded = expandedDimensions(gatedDesc, axis: 1)
            descBroadcast = repeated(expanded, count: embedded.dim(1), axis: 1)
        } else if gatedDesc.dim(1) == 1 {
            descBroadcast = repeated(gatedDesc, count: embedded.dim(1), axis: 1)
        } else {
            descBroadcast = gatedDesc
        }

        return concatenated([embedded, descBroadcast], axis: -1)
    }

    private func normalizeSequence(_ input: MLXArray) -> MLXArray {
        MLXModelUtils.normalizeSequence(input)
    }

    private func sanitizeAux(_ aux: MLXArray) -> MLXArray? {
        MLXModelUtils.sanitizeAux(aux, expectedSize: config.auxSize)
    }
}
