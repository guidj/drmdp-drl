# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning research codebase for **Delayed, Aggregate, and Anonymous Feedback (DAAF)** in MDPs. Implements DFDRL: Deep RL for Delayed Feedback, with reward estimation for policy control in off-policy deep reinforcement learning.

## Workflows

### Post-Change Verification

After **any** code change, run `/verify-change` (see Skills). Do not skip it for small changes.

### Planning Mode

Whenever you create an implementation plan — whether via `/plan`, `EnterPlanMode`, or in response to any non-trivial task request — **always write the plan to `agents/plans/yyyy-mm-dd-{plan-name}.md` before starting implementation**. This is required unconditionally. Use the Write tool to create the file.

### Code Review After Implementation

After completing any non-trivial implementation, run `/post-impl-review` on the touched files (see Skills). Do not rely on self-review of code you just wrote.

## Development Commands

### Environment Setup
```sh
uv venv --python 3.12
make pip-sync  # or: uv sync
```

### Testing
```sh
make test      # Run all tests (tox -e test / pytest)

# Specific test file/class/method
pytest tests/drmdp/control/test_runner.py
pytest tests/drmdp/control/test_runner.py::TestLoadConfigs
pytest tests/drmdp/control/test_runner.py::TestLoadConfigs::test_num_runs_expands_entries

# Pattern matching and coverage
pytest -k "boundary"
pytest --cov=drmdp --cov-report=html  # View: htmlcov/index.html
```

### Linting and Formatting
```sh
make format    # Auto-format (ruff format + ruff check --fix)
make check     # Check formatting/linting/types (ruff + mypy)
make tox       # Run all CI checks
```

## High-Level Architecture

### Core Abstractions (`src/drmdp/`)

**Reward Delay Mechanisms** (`rewdelay.py`):
- `FixedDelay`, `UniformDelay`, `ClippedPoissonDelay`: Delay sampling strategies
- `DelayedRewardWrapper`: Gym wrapper that emits delayed aggregate rewards at window end, `None` otherwise
- `ImputeMissingRewardWrapper`: Replaces `None` rewards with imputed values
- `DataBuffer`: Replay buffer with capacity limits and accumulation modes (first/latest)

**Core Types** (`core.py`):
- `EnvSpec`, `ProblemSpec`, `RunConfig`, `Experiment`, `ExperimentInstance`: Configuration dataclasses
- `EnvMonitorWrapper`/`EnvMonitor`: Track episode returns and steps
- `Seeder`: Deterministic seed generation via Cantor pairing function

### DFDRL: Deep RL for Delayed Feedback (`src/drmdp/dfdrl/`)

Offline reward estimation models (`est_oN.py`) trained on trajectory data:

- **O0**: Baseline — predicts immediate rewards from (state, action) without delay
- **O1**: Window-based — MLP predicts per-step rewards; sum of predictions = observed aggregate reward
- **O2**: Return-grounded — adds regularization loss `MSE(start_return + aggregate, end_return)` to anchor predictions to episodic return progression
- **O3**: EM-based — expectation-maximisation with variable-length padded sequences
- **O4**: Mask-based — learned causal mask (sigmoid / STE / Gumbel) over observations

Each model has a paired `eval_est_oN.py` for evaluation.

### Control: Off-Policy RL with Reward Model Interleaving (`src/drmdp/control/`)

Online policy learning via SB3 SAC with pluggable reward estimation. The policy trains on *estimated* per-step rewards rather than the delayed aggregate rewards emitted by the environment.

**Core abstractions** (`base.py`):
- `Trajectory`: Frozen dataclass — `(observations, actions, env_rewards, terminals, episode_return)`
- `RewardModel`: ABC with `predict(obs, actions, terminals) -> rewards` and `update(trajectories) -> metrics`
- `RelabelingReplayBuffer(ReplayBuffer)`: Calls `reward_model.predict()` inside `sample()` — relabeling is transparent to SAC; stored rewards are never modified

**Reward models**:
- `ircr.py` — `IRCRRewardModel`: non-parametric KNN guidance rewards over a trajectory database; normalised mean episode return of K nearest (s, a) pairs
- `dgra.py` — `DGRARewardModel`: window-based neural reward model; trains on consecutive windows of trajectory data
- `grd.py` — `GRDRewardModel`: graph-based with a causal structure network (`CausalStructure`) and dynamics network; learns which state dimensions causally influence reward

**HC agent** (`hc.py`):
- `HCSAC`: History-Corrected SAC (Han et al., ICML 2022) — learns a decomposed Q-function Q^H(h_t) + Q^C(s_t, a_t) directly from delayed rewards; no separate reward model

**Control loop** (`runner.py`):
- `TrainingArgs`: Frozen configuration dataclass. Required fields include `exp_name: str` (e.g. `"exp-000"`, shared across runs) and `run_id: int` (0-indexed, unique within an experiment) — both are serialised to `experiment-params.json` via `ExperimentLogger`
- `run(args)`: Wraps env → runs SAC or HC to completion → saves model
- `run_batch(configs, mode)`: Unified entry point for all execution. `mode="debug"` runs sequentially; `mode="local"` uses `ProcessPoolExecutor`; `mode="ray"` submits Ray remote tasks
- `RewardModelUpdateCallback`: Tracks episode trajectories, calls `reward_model.update()` at a fixed step interval

**CLI parser design**: `runner.py` uses four independent parsers, each calling `parse_known_args(sys.argv[1:])`:
- `parse_batch_cli` — `--config-file` only
- `parse_exec_args` — `--mode`, `--num-runs`, `--max-workers`, `--ray-address`
- `parse_common_args` — training scalars overrideable from CLI: `--num-steps`, `--sac-*`, `--log-step-frequency`, `--output-dir`
- `parse_single_cli` — all remaining single-run flags (`--env`, `--delay`, `--reward-model-type`, `--update-every-n-steps`, etc.)

**Intended CLI precedence**: `code defaults < config file (any level) < CLI args`. **Known bug**: env-level config file fields for training scalars (e.g. `num_steps`, `sac_learning_rate`) silently override CLI args because `env_defaults = {**global_defaults, **env_fields}` in `_expand_environments` applies env file fields *after* the CLI overrides that were merged into `global_defaults`. The `--num-runs` flag is handled separately via `cli_num_runs` and is not affected.

**Batch config format** (`specs/control-local-batch.json`):
```json
{
  "environments": [
    {
      "env": "MountainCarContinuous-v0",
      "env_kwargs": {"max_episode_steps": 2500},
      "delay": 3,
      "num_steps": 50000,
      "sac_learning_rate": 3e-4,
      "experiments": [
        { "reward_model_type": "ircr", "reward_model_kwargs": {"max_buffer_size": 200, "k_neighbors": 5} },
        { "reward_model_type": "dgra", "reward_model_kwargs": {"train_epochs": 100} }
      ]
    }
  ]
}
```
File-internal precedence: top-level fields → environment-level → experiment-level. `env` is required on every environment entry. `output_dir` is auto-constructed as `<base>/<env>/<exp-NNN>/<run-NNN>`. `num_runs` and `output_dir` are reserved keys excluded from the training-arg merge.

## Skills

Four project-specific skills are available. Use them by name — the skill list in the system prompt shows when they are loaded.

### `/verify-change` — post-change quality gate

Runs `make format && make check && make test`, triages failures, and chains to `/post-impl-review` on success. The single command that replaces the manual post-change workflow.

```
/verify-change                          # auto-detects touched files from git diff
/verify-change src/drmdp/control/grd.py tests/drmdp/control/test_grd.py
```

### `/post-impl-review` — drmdp style and correctness review

Reads every touched file and applies the project's four-area checklist, making corrections directly. Triggered automatically by `/verify-change`; also useful after targeted edits.

```
/post-impl-review src/drmdp/rewdelay.py tests/drmdp/test_rewdelay.py
/post-impl-review src/drmdp/dfdrl/est_o1.py   # episode-boundary check auto-activates
```

### `/add-reward-model <name>` — reward model scaffold

Creates `src/drmdp/control/{name}.py` (implementing `RewardModel` ABC), wires it into `runner.py` (import + `--reward-model-type` choices + `_make_reward_model` branch), and creates `tests/drmdp/control/test_{name}.py` with the minimum three tests. Asks for a plan confirmation before writing any files.

```
/add-reward-model contrastive   # → ContrastiveRewardModel, --reward-model-type contrastive
/add-reward-model oracle
```

### `/research-to-impl <paper-path-or-url>` — paper to reviewed toy implementation

Chains `/summarize-ml-paper` → `/implement-ml-paper-toy` → `/post-impl-review` → `make format && make check && make test`. Asks framework (torch / sklearn) and module name before generating code.

```
/research-to-impl papers/han2022hc.pdf
/research-to-impl https://arxiv.org/abs/2206.15474
```

## Running Experiments

### Single-Run CLI
```sh
# SAC + IRCR
python src/drmdp/control/runner.py --env MountainCarContinuous-v0 \
    --delay 3 --num-steps 50000 --reward-model-type ircr \
    --output-dir /tmp/control-ircr

# HC (no reward model)
python src/drmdp/control/runner.py --env MountainCarContinuous-v0 \
    --delay 3 --agent-type hc --output-dir /tmp/control-hc

# Multiple runs in local parallel mode
python src/drmdp/control/runner.py --env MountainCarContinuous-v0 \
    --delay 3 --num-runs 5 --mode local --output-dir /tmp/out
```

### Batch via JSON Config
```sh
python src/drmdp/control/runner.py --config-file specs/control-local-batch.json \
    --mode local --num-runs 3 --num-steps 50000 --output-dir /tmp/batch-out
```

### Shell Scripts
```sh
sbin/local/control-ircr.sh    # IRCR + SAC
sbin/local/control-dgra.sh    # DGRA + SAC
sbin/local/control-hc.sh      # HC-decomposition SAC
sbin/local/control-delayed.sh # Delayed-rewards baseline (no reward model)
sbin/local/control-batch.sh   # All reward models, both envs
```

## Critical Architectural Patterns

### Episode Boundary Handling

Delayed reward windows MUST NEVER span episode boundaries.

When processing windows:
- Cumulative returns reset at episode starts (no carryover)
- Window generation stops at episode termination
- No data leakage across episodes

**Validation**: `tests/drmdp/dfdrl/test_est_o2.py::TestEpisodeBoundary::test_episode_boundary_no_span`

### Reward Relabeling at Sample Time

`RelabelingReplayBuffer.sample()` calls `reward_model.predict()` on each drawn batch — not at collection time. This means:
- The policy always trains on up-to-date estimates as the model improves
- Stored buffer data is never modified
- Buffer can be reset via `replay_buffer.reset()` after model updates (controlled by `clear_buffer_on_update`)

## Coding Style

Follow **Google's Python Style Guide** with these specific conventions:

**Code Organization (Newspaper Style)**: Classes before module-level functions; within classes, `__init__` first, public methods next, `_`-prefixed helpers last.

**Imports**: Only import modules and types, never bare functions or variables:
```python
# Good
import numpy as np
from typing import List, Dict

# Avoid
from module import function_name
```

**Variable Naming**: Descriptive names, never single letters (`idx`/`jdx` not `i`/`j`).

**Type Annotations**: Always use `typing` module types with full parameterization; prefer immutable (`Mapping`, `Sequence`, `Tuple`) unless mutation is required (`List`, `Dict`).

**Comments**: Explain *why*, not *what*. No meta-commentary (`# CRITICAL:`, `# IMPORTANT:`).

## Testing Guidelines

**Philosophy**: pytest, 80% coverage target. Tests grouped into classes by component; `class TestFoo` per function/class under test.

**Structure**: Tests mirror source — `src/drmdp/control/runner.py` → `tests/drmdp/control/test_runner.py`. Helper functions and fixtures placed *after* all test classes.

**Numerical precision**: `np.testing.assert_allclose(result, expected, atol=1e-6)`

**Stochastic/PyTorch**: Set seeds (`torch.manual_seed(42)`), use `model.eval()` + `torch.no_grad()` for determinism.

**Parameterized tests**: `@pytest.mark.parametrize()` for multiple inputs; `@pytest.fixture` for shared setup.

