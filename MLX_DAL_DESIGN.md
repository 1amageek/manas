# MLX DAL Learning Design

## Purpose
Integrate MLX‑Swift as an optional learner strictly inside the DAL boundary. Manas Core remains non‑learning and deterministic. This design adds no new semantics to Manas.

## Goals
- Improve actuator mapping robustness using local telemetry only.
- Keep safety filters authoritative and always applied.
- Preserve Manas non‑symbolic semantics and deterministic core.
- Keep MLX fully optional and isolated from ManasCore/ManasDAL APIs.

## Non‑Goals
- No learning in Manas Core.
- No access to energies, phases, or raw sensors.
- No world state, simulator internals, or lookahead.
- No discrete mode selection or tokenized actions.

## MLX‑Swift Overview (Research Summary)
- MLX Swift provides `MLXArray` as the central array type plus neural layers, optimizers, random, and FFT utilities.
- Execution is lazy by default; operations are materialized via `eval()` / `asyncEval()`.
- Function transforms include `grad`, `valueAndGrad`, `compile`, and `vmap`.
- Unified memory is available on Apple platforms; device/stream selection is supported.

## Non‑Negotiable Constraints
- Manas Core never learns and never consumes MLX outputs.
- DAL inputs are **drive intents + actuator‑local telemetry only**.
- DAL outputs are actuator commands only.
- Hard safety filters (saturation + rate limits) always take precedence.
- Learning must not access E_i, phi_k, raw sensors, or world state.

## Module Layout (SPM)
- `ManasCore`: protocol semantics, deterministic control.
- `ManasDAL`: safety filters, DAL boundary types, learning policy protocol.
- `ManasMLX`: MLX‑Swift implementation of the DAL learning policy.

`ManasDAL` depends only on protocols. `ManasMLX` is optional and isolated.

## MLX Products (SPM)
The MLX‑Swift package provides products used by `ManasMLX`:
- `MLX` (core tensor ops)
- `MLXNN` (neural layers)
- `MLXOptimizers`
- `MLXRandom`
- `MLXFFT` (optional)
The C/C++ bridge (`Cmlx`) exists but is not required directly by Manas.

## Data Flow
```
Manas Core -> DriveIntents
              |
              v
        SafetyFilter (hard)
              |
              v
     Optional MLX Policy (DAL)
              |
              v
      Actuator Commands
```

## Alignment with Manas Design Philosophy
| Manas Requirement | How the DAL Learner Satisfies It |
| --- | --- |
| Learning only in DAL | MLX policy is isolated inside DAL; Manas Core remains non‑learning. |
| Non‑symbolic semantics | Inputs/outputs are continuous (drives + actuator‑local telemetry); no tokens/modes. |
| Safety precedence | Hard safety filters clamp all outputs before/after MLX deltas. |
| Bounded outputs | DAL outputs are bounded by declared actuator limits. |
| Deterministic core | Learning cannot influence Manas Core; only actuator mapping is adjusted. |

## Interfaces (Conceptual)
### DALLearningPolicy
- `infer(input) -> DALDelta`
- `update(batch) -> LearningUpdateResult`
- `report` (update period, delta/derivative limits, device)

### DALInput
- `drives: [DriveIntent]`
- `telemetry: [ActuatorTelemetry]`

### DALDelta
- `commandDeltas: [ActuatorCommandDelta]` (per actuator)

### LearningReport
- `updatePeriod`, `maxDeltaNorm`, `maxDerivativeNorm`, `device`, `enabled`

## Data Contracts
- **DriveIntent**: bounded activation per actuator index.
- **ActuatorTelemetry**: rpm/current/voltage/temperature per actuator.
- **Normalization**: each feature has a declared reference range; all inputs are normalized to [-1, 1].
- **Output**: MLX outputs **delta only**, scaled by `deltaMax` and clamped before application.

## Recommended Model Architectures
### A) Per‑Actuator Affine (Baseline)
```
u_i = clamp(k_d * drive_i + k_t * telemetry_i + b)
```
- Lowest risk; few parameters; easy to audit.
- Best default for initial integration and safety validation.

### B) Small Per‑Actuator MLP (Optional)
```
x = [drive_i, telemetry_i...]
h = relu(W1 * x + b1)
delta = tanh(W2 * h + b2) * deltaMax
u_i = clamp(u_base + delta)
```
- Outputs are **deltas** only; safety filters remain authoritative.
- Small hidden size only (e.g., 8–16) to reduce mode‑like behavior.

## Execution & Determinism Strategy
- MLX is lazy; **explicit evaluation** is required per control step.
- Inference is synchronous in the control loop.
- Training updates are rate‑limited and logged.
- Device selection is explicit (CPU/GPU) and declared in reports.

## Training Strategy (DAL Only)
- Loss: L2 on actuator command error; optional regularization on delta magnitude.
- Updates: period‑gated; bounded parameter updates; safety filters always override outputs.
- If any safety constraint is violated, updates are skipped and logged.

## Configuration & Logging
- Configuration is optional and applied only when explicitly provided.
- Recommended keys:
  - `MANAS_DAL_LEARNING_ENABLED` (true/false)
  - `MANAS_DAL_UPDATE_PERIOD_MS`
  - `MANAS_DAL_MAX_DELTA_NORM`
  - `MANAS_DAL_MAX_DERIVATIVE_NORM`
  - `MANAS_DAL_DEVICE` (cpu/gpu)
- Logging uses swift‑log. Log enable/disable, update cadence, and safety filter interventions.

## Error Handling
- Errors are typed (e.g., `learningDisabled`, `invalidTelemetry`, `updateTooFrequent`, `deltaLimitExceeded`).
- No silent failures; all errors are logged and surfaced to caller.

## Validation & Profile Reporting
- MLX‑enabled runs must be reported as a distinct profile/badge in conformance/validation reports.
- Default behavior remains MLX‑disabled.

## Platform Support
- MLX‑Swift minimums: macOS 13.3, iOS 16.0, visionOS 1.0.
- This repository targets macOS 26+, so MLX‑Swift is compatible.

## Versioning & Build Notes
- MLX‑Swift is consumed via SPM with a pinned version (track latest published via Swift Package Index).
- Running command‑line tools outside Xcode may require `DYLD_FRAMEWORK_PATH` so Metal shaders load correctly.

## Benchmark Results
Benchmark tests are defined in:
- `manas/Tests/manasTests/MLX/MLXAffineBenchTests.swift`
- `manas/Tests/manasTests/MLX/MLXMLPBenchTests.swift`

### CLI Attempt (Failed)
```
swift test --filter Bench
```
Result (2026-02-02): MLX runtime error loading the default metallib (`Failed to load the default metallib`).
This is expected when running SwiftPM CLI without the metallib bundle.

### Required Run Method
Run the benchmarks via Xcode or xcodebuild so MLX’s metallib is bundled.
Record the printed `[Benchmark]` lines here once executed.

## Tests (Minimum)
- Enforce DAL boundary (no forbidden signals reach MLX policy).
- Enforce update period and delta/derivative limits.
- Enforce safety filters override MLX output.
- Verify deterministic output with learning disabled.
- Verify bounded outputs under extreme telemetry ranges.
