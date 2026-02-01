# Manas Specification (Protocol)

## Scope
Manas defines a non‑symbolic, continuous control protocol with energy/phase semantics, deterministic supervision, reflex precedence, bounded outputs, and a strict DAL learning boundary. It does **not** define any plant or simulation.

## Logging & Configuration
Manas integrates `swift-log` and `swift-configuration` for runtime controls.
Configuration is optional and only applied when explicitly provided.
Environment keys:
- `MANAS_LOG_LEVEL` (trace/debug/info/notice/warning/error/critical; default: info)
- `MANAS_LOG_LABEL` (default: manas)

Runtime bootstrap uses `ManasRuntime` (or `ManasConfigLoader` with a `ConfigReader`) to load configuration and create a logger.

## Non‑Symbolic Constraint (Testable)
Forbidden:
- Any discrete command tokens/labels in control semantics.
- Behaviors that decode a finite action set from continuous inputs.

Required:
- **Continuity**: L2 and L∞ bounds between normalized inputs and outputs at controller steps.
- **Total variation**: bounded TV of outputs over time windows.
- **Operating envelope**: all tests are valid only within declared OED bounds.

Conformance suite must include input families:
step, ramp, PRBS (low‑pass shaped), chirp, and band‑limited noise.

## Inputs, Outputs, and Identifiers
- Inputs: **EnergyState** (E_i ≥ 0) and **PhaseState** (φ_k continuous).
- Identifiers: numeric indices only (UInt32 recommended).
- Outputs: **DriveIntent** (driveIndex, activation), bounded by declared limits.

## Anti‑Token Rules for Phase
Phase signals must avoid discrete collapse:
- Snapping detection and minimum variance under excitation.
- Bandwidth limit B_phi must be declared and enforced.

## Manas Core (Deterministic, No Learning)
Required behaviors:
- Compute E_total and regimes (Normal vs SurvivalOnly).
- Reflex triggers on energy or gradient thresholds.
- Drives synthesized as continuous mappings from (E, φ, regime).
- Apply global and local inhibition; enforce boundedness.

## DAL Boundary (Hard Safety Gate)
DAL inputs: drive intents + actuator‑local telemetry only.
DAL outputs: actuator commands.
Learning is allowed **only** inside DAL, and must not access E_i, φ_k, or sensors.
Hard safety filters (saturation/rate limits) always take precedence.

## Consciousness Interface (Non‑Control)
Hints only: bounded, low‑bandwidth, time‑decaying, and removable without loss of stability.

## Profiles / Badges
Conformance claims must declare:
- B0 Manas‑Baseline (required)
- B1 Manas‑Profile IMU6 Fixed Mapping (for M1)
- B2 Manas‑Strict Permutation Robust (optional)
