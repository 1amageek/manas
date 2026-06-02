# Manas MLX Model Specification (Design-Aligned)

## Purpose
Define the MLX model contract for Manas aligned with `MANAS_NERVE_NETWORK_DESIGN.md`.
This document specifies Core/Reflex model behavior, training boundaries, and runtime
mapping constraints.

## Baseline Priority
- `MANAS_NERVE_NETWORK_DESIGN.md` is the architecture baseline.
- This file defines MLX implementation constraints and must stay consistent with it.
- `MODEL_BOUNDARY_GUIDE.md` defines the boundary between generic Manas model
  logic and robot-specific descriptor/codec/profile specialization.

## Module Boundaries
- **ManasCore**: protocol layers and deterministic contracts.
- **ManasRuntime**: configuration and logging for runtime execution.
- **ManasMLXModels**: model definitions (Core/Reflex/LoRA/shared encoder-decoder).
- **ManasMLXRuntime**: adapters to `CoreController` and `ReflexController`.
- **ManasMLXTraining**: training loops, losses, and optimization.

## Inputs / Outputs
- **Inputs**:
  - ascending channels (sensor-derived streams),
  - optional descending channels (planner/context bias; typed scalar modulation),
  - optional morphology/context vectors.
- **Core output**: bounded command activations interpreted as `DriveIntent`.
- **Reflex output**: bounded non-overwriting corrections (`clamp`, `damping`, `delta`).
- **Final actuator values** are obtained through MotorNerve protocol mapping.
- **Upward feedback contract**: runtime MUST support exporting salience/risk/uncertainty
  summaries derived from internal control state for conscious-layer consumers.
  - Minimum summary fields: `salience`, `risk`, `uncertainty`, `constraintPressure`,
    and `recoveryState`.

## Core Architecture
### Channel-Agnostic Path (Preferred)
- Shared channel encoding with type embeddings for ascending + descending channels.
- Pool/token integration over variable channel counts.
- Core state model:
  - default: pooled features + dual-GRU (fast + slow),
  - optional: attention-based token integration for richer morphologies/tasks.
- Optional adaptation module producing morphology/environment latent factors.
- Shared actuator decoder emits per-channel command activations.

### Baseline Configuration (Normative Default)
Unless a benchmark profile explicitly overrides, MLX runtime uses:
- Core variant: pooled token integration + dual-GRU.
- `typeEmbeddingDim = 16`.
- Descending normalization: descriptor-range normalization when available,
  otherwise bounded running normalization with clip `[-1, 1]`.
- Maximum descending channel count: `64` (higher counts require upstream compression).
- Runtime channel-drop handling: fixed catalog + validity mask with neutral-bias decay.

### Compatibility Path
- Legacy fixed-input core path is allowed for backward compatibility.
- New model configurations must prefer descriptor-driven typed mode.

## Reflex Architecture
- Hybrid reflex mode is supported:
  - analytical stabilization component (e.g., PD-like damping),
  - learned residual component with strict clipping/bounds.
- Reflex remains non-overwriting and must not bypass MotorNerve boundary semantics.
- Reflex safety stabilization takes precedence over conflicting descending bias at
  emergency timescales.

## Learning and LoRA Boundaries
- Primary learning location is Core/Reflex model parameters.
- LoRA adaptation is allowed for Core/Reflex model submodules, including shared
  encoder/decoder components.
- MotorNerve protocol boundary remains explicit and deterministic even when output
  semantics are actuator-typed for a profile.

## Training Phases (Reference)
1. Interface and descriptor alignment (ascending/descending/type maps).
2. Teacher-assisted warm start (BC/IL baseline).
3. Domain randomization and robustness training.
4. Sim-real co-training with constrained adaptation (LoRA-focused).
5. Optional imagination/RL fine-tuning after stability gates are met.

## Safety and Runtime Invariants
- Multi-rate contract must hold (Reflex faster than Core).
- Reflex-safe gating must be preserved.
- Outputs must remain finite, bounded, and descriptor-consistent.
- Runtime adapters must map MLX outputs into `DriveIntent` + `ReflexCorrection`
  before MotorNerve application.
- Descending channels are interpreted as modulation/bias signals; low-level safety
  control authority remains inside Manas runtime loops.
- Runtime arbitration must apply Reflex correction and stabilization precedence
  before MotorNerve mapping when descending bias conflicts with immediate safety.

## Build Note
MLX Swift relies on Metal resources; for training and GPU execution, use Xcode/xcodebuild.
SwiftPM CLI builds are valid for integration and contract verification.
