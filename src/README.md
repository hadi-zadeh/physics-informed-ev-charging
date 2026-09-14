# Source Code Architecture (`src/`)

This directory contains the modular, publication-ready implementation of the physics-informed deep imitation learning framework:

```
src/
├── models/
│   ├── lstm_policy.py          # LSTMPolicy (216,321 parameters, 1-layer LSTM + 4-layer MLP)
│   ├── attention_policy.py     # AttentionPolicy (MHSA Temporal Encoder + MLP Head)
│   └── __init__.py
├── training/
│   ├── losses.py               # CompositeImitationLoss (BC + Tariff + Boundary penalty)
│   ├── trainer.py              # Training loop, optimizer, scheduling, evaluation
│   └── __init__.py
├── expert/
│   ├── lp_expert.py            # Clairvoyant LP Expert solver (Python-MIP / COIN-OR CBC / Clp)
│   └── __init__.py
├── preprocessing/
│   ├── data_pipeline.py        # Physical bounds normalization & rolling horizon windowing
│   └── __init__.py
├── spp/
│   ├── safety_projection.py    # Deterministic 3-Step Safety Post-Processing (SPP)
│   └── __init__.py
└── __init__.py
```

## Key Invariants
- **Zero Hardcoded Paths**: All modules use relative imports and standard PyTorch / NumPy interfaces.
- **Strict Boundary Decoupling**: Differentiable soft penalty in `losses.py` guides policy learning during backpropagation; hard operational constraints and 100% departure SoC fulfillment are deterministically guaranteed by `safety_projection.py`.
- **Exact Continuous LP**: Theorem 1 proves exactness of continuous LP relaxation via COIN-OR CBC / Clp simplex solver (`tolerance=1e-7`, `seed=0`).
