# Manas MLX Model Specification (M1‑ATT)

## Purpose
Define the **learnable model architecture** for Manas using MLX Swift.
This document covers **Core/Reflex model structure** and training phases for M1
(quadcopter attitude stabilization with same‑type swappability).

## Module Boundaries
- **ManasCore**: protocol layers + DSP (NerveBundle/Gating/Trunks/MotorNerve).
- **ManasRuntime**: config/logging for inference.
- **ManasMLXModels**: MLX model definitions + weight IO.
- **ManasMLXTraining**: training loop, losses, optimizers (not required for model delivery).

## MLX Build Note
MLX Swift relies on Metal resources; for training and GPU execution, build and run
through **Xcode/xcodebuild**. SwiftPM CLI builds are supported for library integration
but may not include Metal resources for runtime execution.

## Inputs / Outputs
- **Inputs**: Trunks (Energy / Phase / Quality / Spike) at Core rate.
- **Outputs**: DriveIntent (continuous, bounded **primitive activations**) + Reflex corrections.
- **Reflex output** is **non‑overwriting** (clamp/damping/micro‑intent only).
DriveIntent semantics are defined by the **body/MotorNerve descriptor** for the target robot.

## Core Architecture (M1 Reference)
- **Encoder**: 2‑layer MLP → embedding `e_t`.
- **Fast state**: `GRU_fast(z_fast, e_t)` (100–400 Hz).
- **Slow state**: `GRU_slow(c_slow, pool(e_t))` (5–20 Hz).
- **Policy head**: DriveIntent from `[z_fast, c_slow]`.
- **Aux head (recommended)**: predict next Trunks / next ω for swap adaptation.

## Reflex Architecture (M1 Reference)
- **Ensemble of small MLPs** (8–16 units).
- Inputs: fast NerveBundle outputs (spike/high‑pass, vibration).
- Outputs: clamp gains, damping terms, micro‑intent (bounded).

## Losses (Training)
- **BC loss**: stabilize attitude from baseline controller traces.
- **Aux prediction loss**: next Trunks / next ω (swap adaptation).
- **Reflex regularization**: avoid excessive clamp/damping.
- **Safety penalties**: saturation, rate limit violations.

## Training Phases
1. **BC warm‑start**: Core learns stable control from PID/LQR traces.
2. **Swap adaptation**: training with sensor/actuator swap events.
3. **Reflex HF stress**: impulse/vibration/glitch events.
4. **Optional RL fine‑tuning**: performance improvements only after stability.

## Model Size Guidance (M1)
- Encoder: 128–256
- GRU_fast: 256–512
- GRU_slow: 128–256
- Reflex MLP: 64→64, ×8–16

## Deliverables
- MLX model checkpoint (Core + Reflex).
- Training config (suite IDs, seeds, swap ranges, HF set).
- Evaluation report with recovery/overshoot/HF scores.

## Runtime Integration
- **ManasMLXRuntime** bridges MLX models to `CoreController` and `ReflexController`.
- DriveIntent and Reflex corrections are produced from MLX outputs with bounded mapping.

## Dataset Ingestion
- `ManasTrainingDataset` loads `meta.json` + `records.jsonl` exported by Kuyu.
- `ManasTrunkPipeline` converts sensor samples into Trunks for MLX training.
