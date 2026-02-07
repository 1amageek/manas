import MLX
import MLXNN
import Foundation

public enum LoRAAdapter {
    public enum AdapterError: Error, Equatable {
        case incompatibleModel
        case fileNotFound
    }

    public static func applyToCore(model: ManasMLXCore, config: LoRAConfig) -> ManasMLXLoRACore {
        ManasMLXLoRACore(from: model, loraConfig: config)
    }

    public static func applyToReflex(model: ManasMLXReflex, config: LoRAConfig) -> ManasMLXLoRAReflex {
        ManasMLXLoRAReflex(from: model, loraConfig: config)
    }

    public static func mergeCore(_ loraCore: ManasMLXLoRACore) -> ManasMLXCore {
        let config = loraCore.coreConfig
        let merged = ManasMLXCore(config: config)
        let encoder1Merged = loraCore.encoder1.merged()
        let encoder2Merged = loraCore.encoder2.merged()
        let driveHeadMerged = loraCore.driveHead.merged()
        var modules = ModuleChildren()
        modules["encoder1"] = .value(encoder1Merged)
        modules["encoder2"] = .value(encoder2Merged)
        modules["driveHead"] = .value(driveHeadMerged)
        merged.update(modules: modules)
        return merged
    }

    public static func mergeReflex(_ loraReflex: ManasMLXLoRAReflex) -> ManasMLXReflex {
        let config = loraReflex.reflexConfig
        let merged = ManasMLXReflex(config: config)
        let clampMerged = loraReflex.clampHead.merged()
        let dampingMerged = loraReflex.dampingHead.merged()
        let deltaMerged = loraReflex.deltaHead.merged()
        var modules = ModuleChildren()
        modules["clampHead"] = .value(clampMerged)
        modules["dampingHead"] = .value(dampingMerged)
        modules["deltaHead"] = .value(deltaMerged)
        merged.update(modules: modules)
        return merged
    }

    public static func applyToNerveCore(model: ManasMLXCore, config: LoRAConfig) -> ManasMLXNerveLoRACore {
        ManasMLXNerveLoRACore(from: model, loraConfig: config)
    }

    public static func mergeNerveCore(_ loraCore: ManasMLXNerveLoRACore) -> ManasMLXCore {
        let config = loraCore.coreConfig
        let merged = ManasMLXCore(config: config)

        // Copy all base model parameters (GRU, posteriorNet, etc.)
        merged.update(parameters: loraCore.base.parameters())

        // Merge decoder LoRA weights into sharedActuatorDecoder
        if let l1 = loraCore.sharedDecoderLinear1, let l2 = loraCore.sharedDecoderLinear2 {
            var decoderModules = ModuleChildren()
            decoderModules["linear1"] = .value(l1.merged())
            decoderModules["linear2"] = .value(l2.merged())
            merged.sharedActuatorDecoder?.update(modules: decoderModules)
        }

        // Merge encoder LoRA weights into sharedEncoder
        if let l1 = loraCore.sharedEncoderLinear1, let l2 = loraCore.sharedEncoderLinear2 {
            var encoderModules = ModuleChildren()
            encoderModules["linear1"] = .value(l1.merged())
            encoderModules["linear2"] = .value(l2.merged())
            merged.sharedEncoder?.update(modules: encoderModules)
        }

        return merged
    }

    public static func saveAdapter(_ model: Module, to url: URL) throws {
        let params = model.trainableParameters()
        let flat = params.flattened()
        var dict: [String: MLXArray] = [:]
        for (key, array) in flat {
            dict[key] = array
        }
        try save(arrays: dict, url: url)
    }

    public static func loadAdapter(_ model: Module, from url: URL) throws {
        let loaded = try loadArrays(url: url)
        let nested = ModuleParameters.unflattened(loaded)
        model.update(parameters: nested)
    }
}
