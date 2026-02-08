# manas

A layered CNS-style robotic control system with learnable components.

## Overview

Manas is a general-purpose nerve network for robotic control, inspired by the spinal cord and cerebellum. It functions as an allocator: descending channels (intentions from upper layers) + ascending channels (sensor observations) are processed through a shared encoder to produce per-actuator commands.

### Key Design

- **Channel-agnostic**: Variable N sensors, K intentions, M actuators, same shared weights.
- **Type embeddings**: Ascending and descending channels use the same shared encoder, distinguished by type embeddings.
- **Specialization via LoRA**: Morphology-specific adaptation without architectural changes.
- **Target**: < 1M params base (~540K), iPhone Neural Engine deployment.

### Layer Stack (L0-L6)

```
Sensors (raw streams)
    |
NerveBundle (L1): Normalization, convergence, local transforms
    |
Gating (L2): Continuous reflex-safe gating
    |
Trunks (L3): Abstract feature streams (Energy/Phase/Quality/Spike)
    |
Core (L4): Learnable mid-timescale control -> DriveIntent
    |
Reflex (L5): Learnable HF micro-corrections -> ReflexCorrection
    |
MotorNerve (L6): Actuator mapping -> ActuatorValue
    |
Plant
```

**Learning happens ONLY in Core (L4) and Reflex (L5).** All other layers are fixed transforms.

### Modules

| Module | Dependencies | Description |
|--------|-------------|-------------|
| **ManasCore** | None | Protocol definitions, data types, DSP |
| **ManasRuntime** | swift-log, swift-configuration | Configuration and logging |
| **ManasMLXModels** | ManasCore, mlx-swift | Neural network architectures |
| **ManasMLXRuntime** | ManasCore, ManasMLXModels | MLX-backed controller implementations |
| **ManasMLXTraining** | ManasMLXModels | Training loops, losses, optimizers |

### Neural Architectures

- **ManasMLXCore** — GRU-based (fast+slow) with RSSM world model support.
- **ManasMLXNerveCore** — Shared encoder + type embeddings + descending channels.
- **ManasMLXNerveLoRACore** — LoRA-adapted variant (4 levels: decoder, encoder, GRU, full).
- **ManasMLXHybridReflex** — Analytical PD + NN residual with clip.
- **AdaptationModule** — 1D CNN for online adaptation (rolling history -> z vector).

### Training Pipeline

1. **BC warm-start**: IL from PID/LQR demonstrations with RL co-training (exponential decay).
2. **Domain Randomization + RSSM Imagination RL**: Sim-only with design-based DR.
3. **Co-Training sim+real**: OT alignment, 10-50 real demos, LoRA only.
4. **Deploy**: Remove RSSM heads, keep Core+Reflex+LoRA for iPhone.

## Build

```bash
# SwiftPM (compilation only, no MLX Metal runtime)
swift build

# Xcode (required for MLX Metal resources)
xcodebuild -scheme manas -destination 'platform=macOS'

# Tests
xcodebuild test -scheme manas -destination 'platform=macOS'
```

## Requirements

- Swift 6.2+
- macOS 26+
- MLX Swift 0.29.1+
- Apple Silicon (for MLX runtime)

## Related Packages

- [kuyu](https://github.com/1amageek/kuyu) — Simulation environment for training Manas
- [kuyu-core](https://github.com/1amageek/kuyu-core) — Core simulation protocols
- [kuyu-physics](https://github.com/1amageek/kuyu-physics) — Physics engines
- [kuyu-scenarios](https://github.com/1amageek/kuyu-scenarios) — Evaluation scenarios
- [kuyu-training](https://github.com/1amageek/kuyu-training) — Training data pipeline
- [kuyu-world-model](https://github.com/1amageek/kuyu-world-model) — Learned world model

## License

See repository for license information.
