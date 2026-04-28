# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning research codebase for **Delayed, Aggregate, and Anonymous Feedback (DAAF)** in MDPs. Implements DFDRL: Deep RL for Delayed Feedback, with reward estimation for policy control in off-policy deep reinforcement learning.

## Workflows

### Post-Change Verification

After **any** code change, run:

```sh
make format && make check && make test
```

Identify all issues found. If the issues are minor (formatting, a missing annotation, a small logic fix), address them directly. If they are moderate — for example, they require changes to interfaces, public APIs, or multiple files — propose a plan for review before proceeding.

### Planning Mode
Whenever you create an implementation plan — whether via `/plan`, `EnterPlanMode`, or in response to any non-trivial task request — **always write the plan to `agents/plans/yyyy-mm-dd-{plan-name}.md` before starting implementation**. This is required unconditionally; do not defer it or skip it because the task seems small. Use the Write tool to create the file.

### Code Review After Implementation

After completing any non-trivial implementation — new source files, refactors, or new/modified test files — spawn a dedicated subagent to audit the touched files. Do **not** rely on your own review of changes you just wrote; a fresh subagent catches violations the authoring agent overlooks.

The subagent must check every file that was touched or created across four areas, and make corrections directly rather than just reporting issues:

**Style compliance** (`src/` and `tests/`):
- Newspaper order: classes before module-level functions; within each class, `__init__` first, then public methods, then `_`-prefixed helpers last
- All imports at module level — no inline imports inside functions or methods
- Only modules/types imported, never bare functions or variables (`from module import function` is forbidden)
- No single-letter variable names; no what-comments (comments must explain *why*, not *what*)
- Fully-parameterised `typing` module annotations; mutable types (`List`, `Dict`) only when mutation is required

**Test file structure** (`tests/`):
- Every test is a method inside a class (`class TestFoo`) — no bare module-level `def test_*` functions
- Helper functions and fixtures appear **after** all test classes, never before them
- No inline imports; all imports at module top

**Logical correctness and consistency**:
- Tests must exercise the actual production code under test — not reimplement its logic independently. A test that rebuilds the formula it's supposed to verify is testing nothing. If a function's internals are untestable, refactor the function (e.g. add a callback, return intermediate values) rather than testing a copy of it.
- Check for dead or unreachable code paths introduced by the change
- Verify that related components affected by a refactor remain consistent (e.g. callers updated, related tests still valid)
- Flag any new public API without a corresponding test

**Testability and maintainability**:
- If any logic is only testable by reimplementing it in the test, propose a refactor to expose it (callback pattern, intermediate return value, etc.)
- Flag overly large functions that mix concerns; suggest decomposition if it improves testability
- Identify duplicated logic across source or test files that should be extracted

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

**CLI unification**: Both single-run and batch-run paths converge on `run_batch()`. Global flags `--mode`, `--num-runs`, `--output-dir` apply to both. Use `--config-file` for batch JSON configs; omit it for single-run via CLI args.

**Batch config format** (`specs/control-local-batch.json`):
```json
{
  "output_dir": "...",
  "num_runs": 3,
  "environments": [
    {
      "env": "MountainCarContinuous-v0",
      "delay": 3,
      "experiments": [
        { "reward_model_type": "ircr", "reward_model_kwargs": {...} },
        { "reward_model_type": "dgra", "reward_model_kwargs": {...} }
      ]
    }
  ]
}
```
Field precedence: top-level defaults → environment-level → experiment-level. `env` is required on every environment entry. `output_dir` is auto-constructed as `<base>/<env>/<exp-NNN>/<run-NNN>`.

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
    --mode local --output-dir /tmp/batch-out
```

### Shell Scripts
```sh
sbin/local/control-ircr.sh    # IRCR + SAC
sbin/local/control-dgra.sh    # DGRA + SAC
sbin/local/control-hc.sh      # HC-decomposition SAC
sbin/local/control-delayed.sh # Delayed-rewards baseline (no reward model)
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

## Configuration

**pytest** (`pyproject.toml`):
```toml
addopts = "-v --cov=drmdp --cov-report=xml:cobertura/coverage.xml --cov-report=term-missing"
testpaths = ["tests"]
```

**tox** (`tox.ini`): Environments `test`, `check-formatting`, `check-lint`, `check-lint-types`. Python 3.11/3.12, uv for dependency management.

## Project Structure

```
src/drmdp/
├── core.py, rewdelay.py, optsol.py   # Core abstractions
├── dataproc.py, mathutils.py          # Data & utilities
├── logger.py, metrics.py, ray_utils.py
├── dfdrl/
│   ├── est_o0..o4.py                  # Reward estimation models (offline)
│   └── eval_est_o0..o4.py             # Evaluation scripts
└── control/
    ├── base.py      # Trajectory, RewardModel ABC, RelabelingReplayBuffer
    ├── ircr.py      # Non-parametric KNN guidance rewards
    ├── dgra.py      # Window-based neural reward model
    ├── grd.py       # Graph-based causal reward model
    ├── hc.py        # History-Corrected SAC (Han et al.)
    └── runner.py    # TrainingArgs, run(), run_batch(), CLI

tests/drmdp/           # Mirrors src/ structure
specs/                 # Batch config JSON files
sbin/                  # Experiment scripts (local/, rconfig/, rjobs/)
agents/plans/          # Implementation plans (yyyy-mm-dd-{name}.md)
```
