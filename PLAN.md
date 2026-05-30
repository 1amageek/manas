# Manas Implementation Plan

## 目的
学習可能な CNS‑style 制御プロトコルを実装し、同種交換に順応する。

## 仕様参照
- 仕様要件は `manas/SPEC.md`（および上位 `SPEC.md`）を参照。
- この文書は実装計画のみを扱う（非仕様）。

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

### 6) Robotics + NN 統合
- Descriptor 駆動の型埋め込み（ascending/descending/drive/actuator）
- 非同期サンプル時刻整合（遅延・欠損耐性、有限値保証）
- Multi-rate スケジューラの堅牢化（jitter 下で hold 保証）
- Telemetry フィードバック経路（actuator/safety trace の利用可能化）
- LoRA 適応境界の明確化（Core/Reflex パラメータ限定 + rollback 手順）
- 実行トレース標準化（Bundle→MotorNerve のステップ再構成可能ログ）

### 7) 意識/無意識 双方向I/F 実装
- Descending bias チャネルの型カタログ実装（goal/priority/inhibition/context）
- Upward summary 出力（salience/risk/uncertainty/constraintPressure/recoveryState）
- Arbitration 順序の固定化（Core hold → Reflex補正 → 安全優先解決 → MotorNerve）
- 欠損 descending 時の安定 hold 動作を自動テストで保証

### 8) Conformance ギャップ解消
- Bidirectional flow / arbitration の専用テストを追加
- Upward summary 契約テストを追加
- Latency budget 契約と violation ログの検証を追加
- A4 の Missing 行を段階的に Covered へ更新

## 受け入れ条件
- `manas/SPEC.md` の要件を満たす
- Kuyu M1‑ATT で安定回復を示す
- Robotics + NN 必須メカニズムの conformance を A1/A4 で検証可能

## 実装優先順
1. **契約固定**: 各レイヤ入出力の次元、境界、有限性を固定。
2. **NerveBundle 必須責務**: NB1–NB6 を満たす実装。
3. **Gating reflex‑safe**: fast path 非ゲート化の保証。
4. **Trunks 定義固定**: Energy/Phase/Quality/Spike の次元不変。
5. **Reflex 非上書き補正**: clamp/damping/delta の境界保証。
6. **Core bounded + hold**: 出力境界と未更新時の保持。
7. **MotorNerve 写像**: primitive 活性を actuator 値へ連続写像。
8. **Multi‑rate 実装**: Reflex/Core の周期差を強制。
9. **EmbodimentContract 統合**: 型・意味論を制御境界の contract 起点で解決。
10. **Telemetry 活用**: 制御系が必要な実機状態を入力として利用可能。
11. **再現性固定**: seed/hash/model/embodiment が常に実験成果物へ紐づく。
12. **双方向I/F 完了**: descending bias と upward summary の契約項目を満たす。
13. **安全優先 arbitration**: 競合時に reflex 優先が常に成立する。
