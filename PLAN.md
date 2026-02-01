# Manas Implementation Plan

## 目的
非シンボリック連続制御プロトコルの仕様準拠実装と、適合テストスイートの提供。

## スコープ境界
- **含む**: 非シンボル制約、エネルギー/フェーズ入出力、Manas Core、DAL 境界、適合テスト。
- **含まない**: シミュレーション、プラント定義、UI、ミドルウェア。

## 主要設計方針
- すべての境界識別子は数値 index のみ。
- 連続性（L2/L∞）と TV 制約は**正規化**後に検証。
- 学習は DAL 内に限定し、外部状態を参照しない。
- 反射は常に非反射出力を優先して上書きできる。

## 実装フェーズ
### 1) 型・境界の定義
- `Signal`, `EnergyState`, `PhaseState`, `DriveIntent`
- `OperatingEnvelope`（OED）
- `DALTelemetry` 型（許可されたテレメトリのみ）

### 2) Manas Core
- `ManasCore`（学習なし、決定論）
- `Inhibition` / `Reflex` / `Regime` の明示的モデル
- 出力の boundedness 強制

### 3) DAL 境界
- 入出力の安全フィルタ（飽和・レート制限）
- 学習は DAL 内の限定 API でのみ許可

### 4) 適合テストスイート
- 入力ファミリ: step / ramp / PRBS / chirp / band-limited noise
- 非シンボル検査: 出力スナップ検出、モード誘導検出
- フェーズ反トークン検査: 分散・帯域制約

## 主要マイルストーン
- **M-1**: 型定義と OED の雛形
- **M-2**: Manas Core + DAL 境界が動作
- **M-3**: 適合テストスイート完成

## 受け入れ条件
- Manas 仕様（Paper 1）の MUST/SHALL を満たす
- B0/B1/B2 バッジの宣言に必要な情報が揃う

