# DR-MDP

Reinforcement learning research codebase for **Delayed, Aggregate, and Anonymous Feedback (DAAF)** in MDPs.

## Env

```sh
uv venv --python 3.12
# will run uv sync
make pip-sync
```

## Run Tests

```sh
make test
```

## DFDRL: Deep RL for Delayed Feedback

The `drmdp.dfdrl` module provides three reward estimation approaches for delayed, aggregate, and anonymous feedback scenarios:

### Estimation Modules

#### O0: Immediate Reward Prediction (Markovian Baseline)
- **Model**: Feedforward MLP
- **Input**: Single (state, action) pair
- **Output**: Immediate reward prediction
- **Use case**: Baseline for comparison with delayed feedback models

#### O1: Delayed Aggregate Reward Estimation
- **Models**: MLP, RNN (LSTM/GRU), or Transformer
- **Input**: Sequence of (state, action) pairs
- **Output**: Per-step reward predictions (summed to match aggregate feedback)
- **Use case**: Learning from delayed aggregate rewards without return information

#### O2: Return-Grounded Dual Prediction
- **Models**: Dual architecture with shared embeddings
  - Reward model: MLP, RNN, or Transformer
  - Return model: Transformer (always)
- **Input**: Consecutive window pairs
- **Output**: Reward predictions + return predictions
- **Loss**: Main loss + ρ₁ (predicted return grounding) + ρ₂ (actual return grounding)
- **Use case**: Leveraging return information to improve reward estimation

### Training

All training runs create versioned timestamped outputs: `outputs/{spec}/{unix_timestamp}/`

#### O0 Training

```bash
# Basic training
python -m drmdp.dfdrl.est_o0 \
    --env MountainCarContinuous-v0 \
    --num-steps 100000 \
    --batch-size 64 \
    --output-dir outputs

# Outputs:
# - outputs/o0/{timestamp}/model_o0.pt
# - outputs/o0/{timestamp}/predictions_o0.json
# - outputs/o0/{timestamp}/metrics_o0.json
# - outputs/o0/{timestamp}/config.json
```

#### O1 Training

```bash
# Train single model (MLP, RNN, or Transformer)
python -m drmdp.dfdrl.est_o1 \
    --env MountainCarContinuous-v0 \
    --model-type rnn \
    --delay 3 \
    --num-steps 100000 \
    --batch-size 64 \
    --output-dir outputs

# Train all three architectures
python -m drmdp.dfdrl.est_o1 \
    --env MountainCarContinuous-v0 \
    --model-type all \
    --delay 3 \
    --num-steps 100000

# Outputs (per model type):
# - outputs/o1/{timestamp}/model_{model_type}.pt
# - outputs/o1/{timestamp}/predictions_{model_type}.json
# - outputs/o1/{timestamp}/metrics_{model_type}.json
# - outputs/o1/{timestamp}/config.json
```

#### O2 Training

O2 uses **two-stage training** for better optimization:
- **Stage 1**: Pre-train GNetwork (return predictor) + shared embedding
- **Stage 2**: Train RNetwork (reward predictor) with frozen GNetwork

```bash
# Basic two-stage training
python -m drmdp.dfdrl.est_o2 \
    --env MountainCarContinuous-v0 \
    --reward-model-type rnn \
    --delay 3 \
    --num-steps 100000 \
    --batch-size 64 \
    --lam 0.5 \
    --xi 0.5 \
    --seed 42 \
    --output-dir outputs

# Advanced: Custom epochs and learning rates per stage
python -m drmdp.dfdrl.est_o2 \
    --env MountainCarContinuous-v0 \
    --reward-model-type rnn \
    --delay 3 \
    --num-steps 100000 \
    --batch-size 64 \
    --stage1-epochs 150 \
    --stage2-epochs 100 \
    --stage1-lr 0.01 \
    --stage2-lr 0.005 \
    --lam 1.0 \
    --xi 0.5 \
    --seed 42 \
    --output-dir outputs

# Parameters:
# --stage1-epochs: Epochs for GNetwork training (default: 100)
# --stage2-epochs: Epochs for RNetwork training (default: 100)
# --stage1-lr: Learning rate for Stage 1 (default: 0.01)
# --stage2-lr: Learning rate for Stage 2 (default: 0.01)
# --lam: Weight for ρ₁ regularizer (grounds predictions on aggregate feedback)
# --xi: Weight for ρ₂ regularizer (grounds predictions on actual returns)
# --seed: Random seed for reproducibility

# Outputs (two-stage structure):
# - outputs/stage1/gnetwork_stage1.pt
# - outputs/stage2/rnetwork_{reward_model_type}_stage2.pt
# - outputs/stage2/model_{reward_model_type}_return.pt (for evaluation)
```

### Evaluation

All evaluation modules support two modes:
1. **Predictions mode**: Display predictions from saved JSON files
2. **Interactive mode**: Run live environment rollouts

#### O0 Evaluation

```bash
# Evaluate from saved predictions
python -m drmdp.dfdrl.eval_est_o0 \
    --model-dir outputs/o0/1709564425 \
    --mode predictions \
    --num-examples 10

# Interactive evaluation with live environment
python -m drmdp.dfdrl.eval_est_o0 \
    --model-dir outputs/o0/1709564425 \
    --mode interactive \
    --env MountainCarContinuous-v0 \
    --num-episodes 5

# Output format:
# ================================================================================
#      Actual Reward |      Predicted Reward |           Error
# ================================================================================
#        -0.12345678 |        -0.13456789 |       0.01111111
# ...
# Mean Absolute Error: 0.01234567
# RMSE: 0.01456789
```

#### O1 Evaluation

```bash
# Evaluate from saved predictions
python -m drmdp.dfdrl.eval_est_o1 \
    --model-dir outputs/o1/1709564430 \
    --model-type rnn \
    --mode predictions \
    --num-examples 10

# Interactive evaluation with delayed rewards
python -m drmdp.dfdrl.eval_est_o1 \
    --model-dir outputs/o1/1709564430 \
    --model-type rnn \
    --mode interactive \
    --env MountainCarContinuous-v0 \
    --delay 3 \
    --num-episodes 5

# Output format:
# ================================================================================
#   Window | Seq Len |      Actual Agg |  Predicted Agg |           Error
# ================================================================================
#        0 |       3 |     -0.35678901 |    -0.36789012 |       0.01110111
# ...
```

#### O2 Evaluation

**Note:** Evaluation auto-detects two-stage checkpoint structure. Point `--model-dir` to the parent directory containing `stage2/`.

```bash
# Evaluate from saved predictions (two-stage checkpoint)
python -m drmdp.dfdrl.eval_est_o2 \
    --model-dir outputs \
    --reward-model-type rnn \
    --mode predictions \
    --num-examples 10

# Interactive evaluation with dual predictions
python -m drmdp.dfdrl.eval_est_o2 \
    --model-dir outputs \
    --reward-model-type rnn \
    --mode interactive \
    --env MountainCarContinuous-v0 \
    --delay 3 \
    --lam 0.5 \
    --xi 0.5 \
    --num-episodes 5

# Legacy checkpoint structure (single-stage, deprecated)
python -m drmdp.dfdrl.eval_est_o2 \
    --model-dir outputs/o2/1709564435 \
    --reward-model-type rnn \
    --mode predictions \
    --num-examples 10

# Output format (includes regularizers ρ₁ and ρ₂):
# ================================================================...
#  Win | PLen | CLen |     ActR |     PrdR | ActGp | PrdGp | ρ₁ | ρ₂
# ================================================================...
#    0 |    0 |    3 | -0.356789 | -0.367890 | 0.0 | -45.678 | ... | ...
# ...
# Average ρ₁: 0.12345678
# Average ρ₂: 0.23456789
```

### Output Directory Structure

```
outputs/
├── o0/                          # Immediate reward prediction
│   ├── 1709564425/
│   │   ├── config.json          # Model config and hyperparameters
│   │   ├── model_o0.pt          # Trained model weights
│   │   ├── predictions_o0.json  # Test set predictions
│   │   └── metrics_o0.json      # Training metrics
│   └── 1709564428/              # Later run (preserved)
│       └── ...
├── o1/                          # Delayed aggregate rewards
│   ├── 1709564430/
│   │   ├── config.json
│   │   ├── model_rnn.pt         # Or model_mlp.pt, model_transformer.pt
│   │   ├── predictions_rnn.json
│   │   └── metrics_rnn.json
│   └── ...
└── o2/                          # Return-grounded dual prediction (two-stage)
    ├── stage1/                  # Stage 1: GNetwork pre-training
    │   └── gnetwork_stage1.pt   # GNetwork + shared embedding
    └── stage2/                  # Stage 2: RNetwork training
        ├── rnetwork_rnn_stage2.pt         # RNetwork only
        └── model_rnn_return.pt            # Dual model (for evaluation)
```

## Bumpversion

```sh
bumpver update --patch
```
