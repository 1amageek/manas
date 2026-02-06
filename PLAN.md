# Manas Implementation Plan

## 目的
学習可能な CNS‑style 制御プロトコルを実装し、同種交換に順応する。

## スコープ境界
- **含む**: NerveBundle / Gating / Trunks / Core / Reflex / MotorNerve
- **含まない**: プラント/シミュレーション/GUI

## モジュール分離
- **ManasCore**: 層構造とデータ型
- **ManasRuntime**: 設定/ログ/実行補助
- **manas**: Core + Runtime の再エクスポート
- 学習モジュールは配布対象外（学習は別ターゲット）

## フェーズ
### 1) レイヤー型とI/O契約
- NerveBundle 出力、Gating 係数、Trunks 定義
- DriveIntent + Reflex correction の合成経路
- MotorNerve の写像境界（primitive → actuator）

### 2) Core / Reflex 最小実装
- Core: mid‑timescale 学習可能ブロック
- Reflex: HF向けマイクロ補正（clamp/damping/micro‑intent）

### 3) 同種交換への順応
- センサー/アクチュエータの変動に対する適応学習
- 測定可能な回復指標をKuyuログと整合

### 4) CMI の骨格（任意）
- 非言語 latent での低帯域 I/O のみ定義

### 5) MLX モデルと学習（別モジュール）
- ManasMLXModels: Core/Reflex モデル定義
- ManasMLXTraining: 損失/最適化/学習ループ
- ManasMLXRuntime: MLX モデルの推論接続

## 受け入れ条件
- 本仕様の MUST を満たす
- Kuyu M1‑ATT で安定回復を示す

## 仕様実装の優先順序
1. **契約固定**: 各レイヤ入出力の次元、境界、有限性を固定。
2. **NerveBundle 必須責務**: NB1–NB6 を満たす実装。
3. **Gating reflex‑safe**: fast path 非ゲート化の保証。
4. **Trunks 定義固定**: Energy/Phase/Quality/Spike の次元不変。
5. **Reflex 非上書き補正**: clamp/damping/delta の境界保証。
6. **Core bounded + hold**: 出力境界と未更新時の保持。
7. **MotorNerve 写像**: primitive 活性を actuator 値へ連続写像。
8. **Multi‑rate 実装**: Reflex/Core の周期差を強制。
