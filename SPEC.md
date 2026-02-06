# Manas Specification

## Definition (Normative)
Manas is a **learnable CNS‑style control protocol** with layered structure:
**NerveBundle → Gating → Trunks → Core + Reflex → MotorNerve → Plant**.
It targets same‑type swappability and stable closed‑loop control.
All layer responsibilities listed below are **mandatory**.

## Design Principle (Normative)
The body/plant is part of the computation. Physical dynamics, morphology, and
embedded compliance are treated as **computational resources** that Manas
learns to exploit. Optimal mappings (MotorNerve and primitives) are therefore
**morphology‑dependent**, and Manas must adapt to each robot’s neural and
physical structure.

## Scope
Includes protocol layers, interfaces, learning boundaries, and safety constraints.
Excludes plant dynamics, simulation mechanics, and UI.

## Module Separation (Normative)
Manas is delivered as separate modules:
- **ManasCore**: protocol layers and data types (NerveBundle/Gating/Trunks/Core/Reflex/MotorNerve).
- **ManasRuntime**: configuration + logging for runtime execution.
- **manas**: umbrella target that re‑exports Core + Runtime for convenience.

Training modules (e.g., MLX optimization) are **not required** for model delivery.
Optional MLX modules:
- **ManasMLXModels**: MLX Core/Reflex models.
- **ManasMLXRuntime**: MLX‑backed Core/Reflex controllers.
- **ManasMLXTraining**: training pipeline (not shipped with model).

## Layered Architecture (Normative)
- **L0 Sensors**: raw multi‑channel streams.
- **L1 NerveBundle**: convergence, normalization, local transforms, routing.
- **L2 Gating**: continuous, reflex‑safe gating of ascending streams (slow path only).
- **L3 Trunks**: abstract streams (Energy / Phase / Quality / Spike).
- **L4 Core**: learnable mid‑timescale control.
- **L5 Reflex**: learnable micro‑controllers for HF stabilization.
- **L6 MotorNerve**: actuator mapping (no hard safety filter).

## Inputs / Outputs
- Inputs: signal streams (not discrete command vocabulary).
- Outputs: **DriveIntent** (continuous, bounded), representing **primitive activations**
  rather than direct actuator values.
- Reflex outputs: clamp/damping/micro‑intent applied before MotorNerve.

## Learning (Normative)
- **Allowed and required** in **Core** and **Reflex** for swappability.
- MotorNerve is **not** a learning location; safety is learned in Core/Reflex.
- MotorNerve parameters are **morphology‑dependent** and may be calibrated per body.

## MLX Model Reference
See `MLX_MODEL_SPEC.md` for the MLX Core/Reflex reference architecture and training phases.

## Safety & Stability
- Safety dominates performance (learned safety, no hard filter).
- Reflex must not be fully blocked by Gating (fast path is ungated).
- DriveIntent is always bounded and fixed‑length.
- CMI is **separable**: Manas must remain stable if CMI is absent or disconnected.
- When present, CMI cannot override Reflex or MotorNerve mapping.

## Signal Contract (Normative)
Manas adheres to the shared signal contract in `SIGNAL_CONTRACT.md`.
All inputs and internal streams must respect finite values, monotonic timestamps,
and catalog indices. Missing samples are represented by absence, not NaN.

## Time Contract (Normative)
Manas adheres to the shared time contract in `TIME_CONTRACT.md`.
Core and Reflex periods must be finite, positive, and Reflex must be faster.
When Core or Reflex does not update, the last output is held.

## CMI (Optional)
Non‑linguistic, low‑bandwidth latent interface for co‑resident reasoning systems.
CMI is optional but **cooperation is allowed** when present; Manas must function without it.
Not required for M1.

## Notes on Neural Tract (Future)
Neural Tract is acknowledged as a future concept only; no conformance rules in this spec.

## Mandatory NerveBundle Responsibilities (Normative)
NerveBundle must implement all of the following:
- **NB1 Spatial convergence**: receptive‑field or channel grouping; produce fixed‑dimensional features.
- **NB2 Nonlinear transduction**: saturation/compression for robustness.
- **NB3 Lateral inhibition**: local contrast or anomaly emphasis.
- **NB4 Gain control / normalization**: divisive normalization with stable statistics.
- **NB5 Temporal filtering**: at least one slow path and one fast path.
- **NB6 Routing**: slow path feeds Core (features), fast path feeds Reflex (fastTaps).
Reflex executes on the NerveBundle fast path and does not require Core to update.

## Fixed NerveBundle Parameters (Normative)
Quality estimation and temporal shaping use fixed constants:
- `qualityFloor = 0.2`
- `transductionGain = 2.0`
- `slowTau = 0.05 s`
- `fastTau = 0.005 s`
- `normalizationTau = 0.2 s`
- `lateralInhibitionStrength = 0.2`
- `delayPenaltyPerSecond = 0.2`
- `missingPenalty = 0.5`
- `deltaPenalty = 0.1`
- `normalizationEpsilon = 1e-6`

## Mandatory Gating Constraints (Normative)
- Gating is **continuous** (no discrete modes).
- Gating is **reflex‑safe**: fast path is never gated off.
- Gate factors are bounded in `[minGate, maxGate]` with `minGate > 0`.

## Multi‑Rate Execution (Normative)
Manas must enforce distinct update periods:
- **Reflex** updates at the fast rate.
- **Core** updates at a slower rate.
When Core does not update, the last DriveIntent is held.

## Reflex Boundary (Normative)
Reflex is a bounded correction layer and must obey strict constraints:
- Corrections only modify existing DriveIntent channels.
- `clampMultiplier` and `damping` must remain in `[0, 1]`; `delta` must be finite.
- Reflex must not create or remove DriveIntent channels.
- Reflex must not implement a hard safety filter or override MotorNerve.

## Trunks (Normative; Minimum Definition)
- **Energy**: non‑negative magnitude of gated features.
- **Phase**: signed/phase‑bearing components of gated features.
- **Quality**: sensor reliability estimates.
- **Spike**: fast‑path magnitude indicators.

## DriveIntent (Normative)
- Fixed length for the target plant.
- Continuous and bounded; all outputs are clamped before MotorNerve.

## Primitive Representation (Normative)
- DriveIntent is a vector of **motor‑primitive activations**.
- DriveIntent may include **optional continuous parameters** per primitive.
- Primitive semantics are defined by the **body/MotorNerve descriptor**, not globally.
- The primitive bank may be **hierarchical** (task‑level → allocation → actuator).
- Primitives are designed to exploit **body dynamics** (inertia, compliance, coupling).

## MotorNerve (Normative)
- The **MotorNerveEndpoint** maps **DriveIntent (+ Reflex corrections)** to actuator values.
- MotorNerve may be **multi‑stage** internally; intermediate stages map MotorNerve signals.
- MotorNerve is **morphology‑dependent** and must preserve boundedness/continuity.
- MotorNerve must not implement a hard safety filter.
