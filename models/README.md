# Neural Policy Architectures & Checkpoints (`models/`)

## Architecture Specifications

### 1. Proposed Policy: `LSTMPolicy` (216,321 Parameters)
- **Input Dimension**: 74
  - 24-step historical spot price series ($/kWh)
  - 24-step historical rooftop solar PV generation (kW)
  - 24-step historical residential baseload demand (kW)
  - Normalized time of day ($t/24$)
  - Normalized battery State of Charge ($\mathrm{SoC}/40$)
- **Temporal Feature Extractor**:
  - 1-layer LSTM: `input_size=3`, `hidden_size=64`, `batch_first=True` ($17,664$ params)
- **Fusion Layer**:
  - Concatenation of final LSTM hidden state (64) + time (1) + SoC (1) = 66 features
- **MLP Decision Head**:
  - `Linear(66, 512)` ($34,304$ params) -> ReLU
  - `Linear(512, 256)` ($131,328$ params) -> ReLU
  - `Linear(256, 128)` ($32,896$ params) -> ReLU
  - `Linear(128, 1)` ($129$ params)
- **Action Bounding**: `Hardtanh(min_val=-7.0, max_val=7.0)`
- **Total Trainable Parameters**:
  $$17,664 + 34,304 + 131,328 + 32,896 + 129 = 216,321$$

### 2. Attention Ablation: `AttentionPolicy` (990,917 Parameters)
- **Temporal Attention Extractor**:
  - Multi-Head Self-Attention: `embed_dim=64`, `num_heads=4`, `dropout=0.1`
  - Position-wise Feed-Forward Network: `Linear(64, 128)` -> ReLU -> `Linear(128, 64)`
  - Flattened output dimension: $24 	imes 64 = 1,538$ (with scalars)
- **MLP Head**: Same hidden structure ($512 ightarrow 256 ightarrow 128 ightarrow 1$).
- **Total Trainable Parameters**: 990,917.

## Pretrained Checkpoints Policy
Model checkpoint binaries (`.pt`) for all five seeds are archived on Zenodo (DOI placeholder: `10.5281/zenodo.XXXXXXX`) to maintain a lightweight GitHub repository footprint (< 25 MB).
