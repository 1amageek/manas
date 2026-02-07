# Manas Nerve Network Design

## Status: Draft (2026-02-07)

General-purpose nerve network architecture for Sim-to-Real robotic control.

---

## 1. Vision

Manas is a **general-purpose nerve network** — analogous to the spinal cord and cerebellum.
It does not decide *where* to go. It decides *how* to realize upper-layer commands
given the current physical state, hardware constraints, and real-time dynamics.

The same network handles drones, humanoids, and manipulators.
Specialization comes from **connection topology and learned weights (LoRA)**,
not from architectural changes.

```
Upper Layer (VLA / Consciousness / Planner)
  "Move this way" → descending channels (intention vectors)
              ↓ DESCENDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Manas (General-Purpose Nerve Network)
  descending + ascending + morphology → actuator commands
  - Real-time allocation
  - Constraint satisfaction (saturation, contact, failure)
  - Disturbance compensation
  - Hardware adaptation (LoRA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ASCENDING ↑            ↓ OUTPUT
Sensors (variable N)   Actuators (variable M)
```

### Biological Analogy

| Biology | Manas |
|---|---|
| Corticospinal tract (descending) | Descending channels (intention vectors from upper layer) |
| Sensory afferents (ascending) | Ascending channels (sensor streams) |
| Spinal cord + cerebellum | Manas Core → allocation + adaptation |
| Motor neurons | Shared decoder → per-actuator commands |
| Muscles | Actuators |
| Synaptic weights | LoRA (morphology-specific adaptation) |

The corticospinal tract is not a single wrench vector. It is a **bundle of nerve fibers**
carrying heterogeneous signals — desired movements, modulation gains, urgency cues.
Manas treats descending channels the same way as ascending (sensor) channels:
each is a typed signal processed by a shared encoder.

---

## 2. Interface Contract

### 2.1 Three Channel Bundles

Manas processes three kinds of channel bundles. All use the same encoding mechanism.

```
Ascending channels  (sensors → Manas):      N channels, variable
Descending channels (upper layer → Manas):   K channels, variable
Output channels     (Manas → actuators):     M channels, variable
```

Every channel is a scalar value + type_embedding. Manas does not distinguish
"this is a sensor" from "this is an intention" at the architectural level.
The type_embedding carries that semantic information.

### 2.2 Ascending Channels (Sensors → Manas)

```
ascending = [s_0, s_1, ..., s_{N-1}]
```

Each channel: `(value, channelIndex, timestamp, type_embedding)`

Examples:
- gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z (IMU)
- joint_angle_0, joint_velocity_0, ... (proprioception)
- contact_force_left, contact_force_right (touch)
- battery_voltage, motor_temperature (system state)

Derived from RobotDescriptor `signals.sensor[]`.

### 2.3 Descending Channels (Upper Layer → Manas)

```
descending = [d_0, d_1, ..., d_{K-1}]
```

Each channel: `(value, type_embedding)`

The upper layer sends whatever it wants. Manas does not prescribe the format.
Examples by upper layer type:

| Upper Layer | Descending Channels | K |
|---|---|---|
| Position PID | desired_thrust, desired_τx, desired_τy, desired_τz | 4 |
| VLA model | task_z_0, task_z_1, ..., task_z_d (latent intention) | d |
| Human teleop | joystick_roll, joystick_pitch, joystick_throttle, joystick_yaw | 4 |
| Walking planner | desired_vx, desired_vy, desired_yaw_rate | 3 |
| None (autonomous) | (empty) | 0 |

When K=0 (no descending signal): Manas maintains current state (hover, stand, hold)
using ascending channels only.

The key insight: **descending channels are processed by the same shared_encoder
as ascending channels.** The type_embedding tells Manas what each signal means.
Manas learns the relationship between intention signals and required actuator responses.

### 2.4 Output Channels (Manas → Actuators)

```
commands = [c_0, c_1, ..., c_{M-1}]   normalized actuator activations
```

Each channel: `(value, actuator_type_embedding)`

The output count M is determined by RobotDescriptor `signals.actuator[]`.
The shared_actuator_decoder produces one scalar per actuator channel.

This is the existing DriveIntent → MotorNerve → ActuatorValue pipeline,
generalized to variable actuator count.

---

## 3. Model Architecture

### 3.1 Design Principles

1. **Channel-agnostic**: N sensors in, M actuators out. Same network weights.
2. **Shared processing**: Per-channel encoder/decoder with shared weights.
3. **LoRA adaptation**: Morphology-specific weight adjustments (~50K params).
4. **Band separation**: NN handles mid/low frequency. Analytical PD handles high frequency.
5. **Small**: Base model < 1M params. Runs on iPhone Neural Engine.

### 3.2 Architecture Overview

```
Inputs:
  ascending[N]   (sensor channels, from RobotDescriptor)
  descending[K]  (intention channels, from upper layer — may be empty)
  morphology[D]  (from RobotDescriptor, precomputed)

┌─────────────────────────────────────────────────────┐
│ Channel Tokenizer (L1-L3 equivalent)                │
│                                                     │
│   For each ascending channel i (sensor):            │
│     token_i = shared_encoder(value_i, type_embed_i) │
│                                                     │
│   For each descending channel j (intention):        │
│     token_j = shared_encoder(value_j, type_embed_j) │
│                                                     │
│   morph_token = morphology_encoder(morphology)      │
│                                                     │
│   tokens = [asc_0, ..., asc_N,                      │
│             desc_0, ..., desc_K,                     │
│             morph]                                   │
│                                                     │
│   Note: ascending and descending use the SAME       │
│   shared_encoder. Type embeddings distinguish them.  │
├─────────────────────────────────────────────────────┤
│ Core (L4 equivalent) — Allocation Network           │
│                                                     │
│   Option A: Attention over all tokens               │
│     features = self_attention(tokens)               │
│     Natural for variable N+K                        │
│                                                     │
│   Option B: Pooling + GRU                           │
│     asc_pool  = pool(ascending_tokens)              │
│     desc_pool = pool(descending_tokens)             │
│     concat = [asc_pool, desc_pool, morph_token]     │
│     features = dual_gru(concat)                     │
│       fast_gru: reactive (100-400Hz)                │
│       slow_gru: adaptive (5-20Hz)                   │
│                                                     │
│   + Adaptation Module                               │
│     z = adaptation(history_100steps)                 │
│     features = [features, z]                        │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Actuator Decoder (L6 equivalent)                    │
│                                                     │
│   For each actuator channel j:                      │
│     command_j = shared_actuator_decoder(            │
│       features, actuator_type_embedding_j           │
│     )                                               │
│                                                     │
│   commands = [c_0, c_1, ..., c_{M-1}]              │
├─────────────────────────────────────────────────────┤
│ Reflex (L5 equivalent) — Hybrid                     │
│                                                     │
│   analytical_pd = K_d * omega_error                 │
│   nn_residual   = clip(residual_net(features), lim) │
│   correction    = analytical_pd + nn_residual       │
│                                                     │
│   final_command = commands + correction              │
└─────────────────────────────────────────────────────┘
```

### 3.3 Core Variants

Both variants are supported. Selection depends on deployment constraints:

**Variant A: Attention-based**
- Self-attention over all tokens (ascending + descending + morphology)
- Natural handling of variable-length input (N+K changes freely)
- Higher compute cost per step
- Better for complex morphologies (humanoid) and rich descending signals (VLA)

**Variant B: Pool + GRU (recommended for first implementation)**
- Pool ascending tokens → fixed-size feature
- Pool descending tokens → fixed-size feature (zero vector if K=0)
- Concatenate with morphology token
- Dual GRU (fast + slow timescale)
- Lower compute, proven on drone (Berkeley paper)
- Adaptation module: 1D CNN over history → z (8-32 dim)

Variant B maps directly to current ManasMLXCore with minimal changes:
- Add descending input (GoalCore's goalEncoder pattern: K=0 → backward compat)
- Add morphology embedding
- Replace fixed inputSize with pooled tokens

### 3.4 Shared Encoder (Ascending + Descending Unified)

```swift
// ONE shared encoder for ALL input channels — ascending AND descending.
// The type_embedding carries the semantic meaning.
shared_encoder(value: Float, type_embed: [Float]) -> [Float]

// Actuator Decoder: same weights for all output channels
shared_actuator_decoder(features: [Float], type_embed: [Float]) -> Float
```

This is the biological neuron principle:
a nerve fiber is generic. Whether it carries proprioception or motor intention
is determined by its **connection** (type), not its **structure** (architecture).

The same shared_encoder processes:
- A gyro reading (ascending)
- A desired thrust command (descending)
- A VLA latent dimension (descending)
- A contact force estimate (ascending)

The type_embedding distinguishes them. The network learns what each type means
in the context of producing actuator commands.

### 3.5 Type Embeddings

Each channel type has a learned embedding vector:

```
Ascending types:   gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z,
                   contact_force, joint_angle, joint_velocity,
                   battery_voltage, motor_temperature, ...

Descending types:  desired_thrust, desired_torque_x, desired_torque_y,
                   desired_velocity_x, desired_yaw_rate,
                   task_latent_0, task_latent_1, ..., task_latent_d,
                   joystick_roll, joystick_pitch, ...

Actuator types:    motor_thrust, joint_torque, gripper_force, ...
```

These embeddings are part of the base model (shared across morphologies).
New channel types can be added by extending the embedding table —
no architectural change required.

Derived from RobotDescriptor `signals.sensor[].type`, `signals.actuator[].type`,
and a new `signals.descending[].type` for upper-layer channels.

### 3.6 Parameter Budget

| Component | Params | Notes |
|---|---|---|
| shared_encoder (2-layer MLP, 128) | ~35K | type_embed(16) + value(1) → 128 → 128 |
| morphology_encoder (linear, 128) | ~5K | morph_descriptor → 128 |
| fast_gru (hidden 256) | ~300K | 3 × (256+384) × 256 |
| slow_gru (hidden 128) | ~100K | 3 × (128+384) × 128 |
| adaptation_module (1D CNN) | ~35K | 100-step history → z(16) |
| shared_actuator_decoder (2-layer MLP) | ~35K | (384+16+16) → 128 → 1 |
| reflex_residual_net (1-layer) | ~5K | minimal |
| type_embeddings (all channel types) | ~10K | ~40 types × 16 dim |
| **Base Total** | **~525K** | **~1.1 MB in fp16** |
| LoRA (r=8, encoder+decoder) | ~50K | per morphology |

Note: wrench_encoder is eliminated. Descending channels (including wrench-like signals)
are processed by the same shared_encoder as ascending channels. This simplifies the
model and reduces parameters.

iPhone Neural Engine inference: < 0.5ms per step.

---

## 4. LoRA Strategy

### 4.1 Role: Morphology-Specific Adaptation

LoRA does NOT change what the network does. It adjusts HOW it does it
for a specific body: inertia, thrust coefficients, latency, friction, sensor characteristics.

Conceptually, LoRA encodes the **latent physical parameters z** of the body.

### 4.2 Application Scope (Progressive)

```
Level 1 (start here):  actuator_decoder only
Level 2 (if needed):   + sensor_encoder
Level 3 (if needed):   + GRU output projection
Level 4 (avoid):       full model LoRA (risk of classifier behavior)
```

Start with Level 1. Only expand if drone/humanoid transfer requires it.

### 4.3 Anti-Pattern: Classifier LoRA (Failure Mode 3)

If LoRA rank is too high or applied to too many layers, the optimizer finds
a shortcut: classify the condition and switch output modes discretely.
This works in-distribution but fails catastrophically out-of-distribution.

Mitigation:
- Keep rank low (r=4-8)
- Constrain LoRA to output continuous z estimates
- DR uses continuous distributions (not discrete conditions)

### 4.4 TinyLoRA Possibility (Future)

Paper finding: RL with 13 params achieves 91% of full fine-tuning.
For swap adaptation (runtime), TinyLoRA-scale updates (r=1, ~10-100 params)
could enable real-time online adaptation on-device.

---

## 5. Training Pipeline

### Overview

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Deploy
Interface   Teacher    DR+RL     Co-Train   iPhone
Definition  Distill    (Sim)     (Sim+Real)
```

### Phase 0: Interface and Morphology Definition

- Define ascending/descending/output channel types and type_embeddings
- Extend RobotDescriptor with `signals.descending[]` for upper-layer channels
- Configure RobotDescriptor for target morphology
- Implement shared_encoder and shared_actuator_decoder
- Implement analytical Reflex (PD damping)

### Phase 1: Teacher Distillation (IL → RL exponential decay)

**Method**: Combined imitation learning + reinforcement learning
(Berkeley drone paper, proven effective)

```
Loss = (1-α) · R_RL(π) - α · L_IL(π)
α = e^(-0.001 · epoch)     // IL fades, RL takes over
```

**Teacher generation**:
- Drone: PID/LQR controller outputs desired thrust/torques as descending channels
- Humanoid: WBC/MPC outputs desired centroidal wrench as descending channels
- Manas learns: given descending intention + ascending sensors, produce actuator commands that match teacher

**Key insight from TinyLoRA paper**: RL outperforms SFT by 100-1000x in low-parameter
regimes. The exponential decay schedule transitions from safe IL startup to efficient RL.

**Data**: Teacher runs in Kuyu, 200+ episodes per morphology (VLA paper threshold).

### Phase 2: Domain Randomization + Imagination RL (Sim-only)

**Domain Randomization** (design-principle-based, Berkeley paper):

```
Size factor c ∈ [0, 1]:
  arm_length  ∝ c           (linear)
  mass        ∝ c³          (volume)
  inertia     ∝ c⁵          (physics)
  drag        ∝ c²          (area)
  thrust_coef: exponential scale

All parameters: ±20% uniform noise
Mid-episode re-randomization: mass, inertia, CoM shift (payload simulation)
```

Integrated with Kuyu's existing mechanisms:
- SwappableActuatorEngine: gain, lag, max_output, deadzone
- HFStressEvent: impulse, vibration, sensor_glitch, actuator_saturation, latency_spike
- TorqueDisturbanceField: external torques

**RSSM Imagination RL** (existing implementation):
- World Model: categorical latent (32×32), KL balancing
- Imagination rollout: horizon 15-30 steps
- Actor-Critic: GAE λ=0.95, γ=0.997
- Used only in sim. Removed at deployment.

**Reward** (extended from current dense reward):

```
r = alive_bonus
  - intention_tracking_weight · ||state_actual - state_desired||²
  - smoothness_weight         · ||a_t - a_{t-1}||²
  - saturation_penalty        · max(0, |a| - saturation_threshold)²
  - effort_weight             · ||a||²
```

Note: "intention tracking" measures how well Manas realizes the descending signal's
intent — for wrench-like signals this is force/torque tracking, for desired-state
signals this is pose/velocity tracking.

### Phase 3: Co-Training with Real Data (Sim + Real)

**Method**: Sim-and-Real co-training (Paper 1) + OT domain alignment (Paper 2)

```
L_total = α · L(θ; D_kuyu) + (1-α) · L(θ; D_real) + β · L_UOT
```

Where L_UOT aligns joint distributions of (observations, actions) across domains:

```
C_ϕ = α₁ · d_Z(f_ϕ(obs_sim), f_ϕ(obs_real))
    + α₂ · d_A(act_sim, act_real)

L_UOT = min_Π ⟨Π, Ĉ_ϕ⟩ + ε·Ω(Π) + τ·KL(Π1||p) + τ·KL(Πᵀ1||q)
```

Unbalanced OT handles the data imbalance (thousands of sim trajectories, 10-50 real demos).

**During co-training: LoRA only.** Base model is frozen.

**Real data collection**:
- 10-50 demonstrations per morphology
- IMU + proprioception logging
- Teacher-operated (teleop or teacher controller on real hardware)

### Phase 4: Deploy

- Remove RSSM heads (reward, continue, value, prior, posterior)
- Keep: sensor_encoder + core (GRU) + actuator_decoder + reflex (analytical PD + NN residual)
- Base (~540K) + LoRA (~50K) → ~1.2 MB fp16
- Target: iPhone Neural Engine, CoreML or MLX
- Control frequency: 100-500Hz (< 0.5ms inference)

---

## 6. Failure Modes and Mitigations

### FM1: Infeasible Intention (Humanoid)

Upper layer sends descending signals that cannot be physically realized due to
contact constraints, friction cones, joint limits.

**Mitigation**:
- Upper layer can include priority/softness as descending channels
- Manas learns to gracefully degrade when intention conflicts with physics
- Training with DR (Phase 2) exposes infeasible regions
- Manas outputs the best feasible approximation

### FM2: Saturation Breakaway (Drone)

Near actuator saturation, small perturbations cause unrecoverable divergence.

**Mitigation**:
- Actuator state (current output level) included as ascending channels
- Saturation penalty in reward (Phase 2)
- DR includes thrust coefficient variation, voltage dropout
- System monitoring can inject headroom_hint as descending channel

### FM3: LoRA Classifier Collapse

LoRA learns discrete condition classification instead of continuous adaptation.
Works in-distribution, fails catastrophically out-of-distribution.

**Mitigation**:
- LoRA rank r=4-8 (low)
- Apply to decoder first, expand only if needed
- DR uses continuous distributions
- Regularize LoRA toward smooth z manifold

### FM4: Bandwidth/Latency Instability

NN tries to control high-frequency dynamics, but real-world latency
erodes phase margin and causes oscillation/divergence.

**Mitigation**:
- **Band separation**: analytical PD handles high frequency
- NN output: rate limited + low-pass filtered
- Residual form: Δw with clipping (safety bound)
- DR includes latency, quantization, actuator dynamics (Phase 2)

### FM5: Contact Estimation Error (Humanoid)

Incorrect contact state estimation leads to wrong actuator allocation → fall.

**Mitigation**:
- contact_confidence as ascending channel (from external estimator)
- Upper layer can send confidence as descending channel
- Conservative allocation learned through DR with contact uncertainty
- DR: slip, partial contact, floor stiffness variation

---

## 7. Relation to Existing Manas Architecture

### Layer Mapping

| Manas Layer | Current Implementation | Nerve Network Design |
|---|---|---|
| L0 Sensors | Fixed 6-ch IMU | Variable N ascending channels |
| — (new) | — | Variable K descending channels (from upper layer) |
| L1 NerveBundle | Imu6NerveBundle (fixed) | shared_encoder (channel-agnostic, asc+desc) |
| L2 Gating | ContinuousGating | Attention / pooling over all tokens |
| L3 Trunks | 4-bank fixed size | Token integration → feature vector |
| L4 Core | ManasMLXCore (fixed I/O) | Core allocation (variable input, variable output) |
| L5 Reflex | NN (clamp/damping/delta) | Analytical PD + NN residual (clipped) |
| L6 MotorNerve | Fixed 4-motor mixer | shared_actuator_decoder (variable M) |

### Protocol Compatibility

Existing Manas protocols (`NerveBundle`, `CoreController`, `ReflexController`)
are already generic. The change is in **MLX model implementations**, not protocols.

### Migration Path

```
Step 1: Add descending channel input to ManasMLXCore
        Reuse GoalCore's goalEncoder pattern: descending pool + gate
        K=0 (no descending) → backward compatible with current behavior

Step 2: Add type_embeddings to sensor/actuator/descending channels
        Extend ManasMLXCoreConfig with type_embedding_dim
        Maintain backward compat: if no embeddings, use fixed identity

Step 3: Implement shared_encoder / shared_actuator_decoder
        Replace fixed Linear(inputSize→embed) with shared encoder
        Replace fixed Linear(state→driveCount) with per-actuator decoder
        Same encoder for ascending + descending channels

Step 4: Reflex hybrid (analytical PD + NN residual)
        Add PD damping as non-learned component
        Wrap existing NN reflex output with clipping

Step 5: Adaptation module (1D CNN or GRU history → z)
        Add to ManasMLXCore as optional component

Step 6: LoRA as morphology adapter
        Existing ManasMLXLoRACore extended with scope control
```

Each step is independently testable. Existing drone tests pass at every step.

### RobotDescriptor Integration

RobotDescriptor already defines:
- `signals.sensor[].type` → ascending type embeddings
- `signals.actuator[].type` → actuator type embeddings
- `sensors[].channels` → ascending channel count N
- `actuators[].channels` → output channel count M
- `motorNerve.stages` → allocation mapping

New addition needed:
- `signals.descending[]` → descending type embeddings and channel definitions
- `control.descendingChannels` → descending channel count K

The gap: MLX models currently ignore RobotDescriptor and use hardcoded dimensions.
Fix: ManasMLXCoreConfig is generated FROM RobotDescriptor at initialization.

---

## 8. References

- [1] Sim-and-Real Co-Training (2503.24361): Co-training recipe, Digital Cousins,
      50 real demos, 38% improvement over real-only
- [2] Generalizable Domain Adaptation (2509.18631, NeurIPS 2025): OT-based joint
      distribution alignment, Unbalanced OT, 10-25 real demos
- [3] Learning-based Quadcopter Controller with Extreme Adaptation (2409.12949):
      IL+RL exponential decay, adaptation module φ (1D CNN→z(8)),
      design-based DR, 16× generalization, 0.165ms inference
- [4] Learning to Reason in 13 Parameters (2602.04118): TinyLoRA, RL >> SFT
      in low-parameter regime (100-1000×), SVD-based weight projection
- [5] Manas SPEC.md: Layered control protocol (L0-L6)
- [6] RobotDescriptor Specification: Morphology-agnostic robot definition
- [7] Signal Contract: Channel-based signal semantics

---

## 9. Open Questions

- [ ] Attention (Variant A) vs Pool+GRU (Variant B) — benchmark on drone first
- [ ] Type embedding dimension (8? 16? 32?)
- [ ] Descending channel normalization (per-type scale? learned? raw?)
- [ ] RSSM imagination horizon for different morphologies
- [ ] Co-training OT loss weight β tuning
- [ ] Minimum real demos per morphology for reliable transfer
- [ ] Online LoRA update on-device (TinyLoRA feasibility on iPhone)
- [ ] Pooling strategy for variable-length channel bundles (mean? attention? weighted?)
- [ ] Maximum number of descending channels before pooling becomes lossy
- [ ] How to handle descending channels appearing/disappearing at runtime
      (e.g., VLA connection drops → K changes from d to 0)
