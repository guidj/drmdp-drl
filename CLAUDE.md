# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning research codebase for **Delayed, Aggregate, and Anonymous Feedback (DAAF)** in MDPs. Implements DFDRL: Deep RL for Delayed Feedback, with reward estimation for policy control in off-policy deep reinforcement learning.

## Workflows

### Planning Mode
When in planning mode, export the created plan to a `plans/` directory using the filename format `yyyy-mm-dd-{plan-name}.md`.

## Development Commands

### Environment Setup
```sh
uv venv --python 3.12
make pip-sync  # or: uv sync
```

### Testing
```sh
# Run all tests
make test  # or: tox -e test, pytest

# Specific test file/class/method
pytest tests/drmdp/test_mathutils.py
pytest tests/drmdp/test_mathutils.py::TestHashtrick
pytest tests/drmdp/test_mathutils.py::TestHashtrick::test_empty_input

# With coverage and pattern matching
pytest -v --cov=drmdp --cov-report=term-missing
pytest -k "boundary"  # Run tests matching "boundary"
pytest --cov=drmdp --cov-report=html  # Generate HTML coverage report
```

### Linting and Formatting
```sh
make check     # Check formatting/linting/types
make format    # Auto-format code
make format-nb # Format notebooks
make tox       # Run all CI checks
bumpver update --patch  # Bump version
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

**Optimization** (`optsol.py`):
- `MultivariateNormal`: Bayesian linear regression and least squares solvers (exact/pseudo inverse)
- Convex optimization via CVXPY with constraint support

### DFDRL: Deep RL for Delayed Feedback (`src/drmdp/dfdrl/`)

Three reward estimation approaches with increasing sophistication:

**O0 - Immediate Reward Baseline** (`est_o0.py`):
- Baseline: predicts immediate rewards without delay
- `RNetwork`: Feedforward MLP for (state, action) → reward

**O1 - Window-Based Reward Estimation** (`est_o1.py`):
- Predicts per-step rewards from delayed aggregate feedback
- `RNetwork`: MLP processing (state, action, term) with polynomial features
- Training: sum of predicted rewards = observed aggregate reward
- Data generation: `delayed_reward_data()` creates windows from trajectories

**O2 - Return-Grounded Reward Prediction** (`est_o2.py`):
- Single `RNetwork` (MLP) with return-consistency regularization
- **Training**: Single stage with combined loss:
  - `reward_loss`: MSE between summed per-step predictions and observed aggregate reward
  - `regu_loss`: MSE(start_return + aggregate_reward, end_return) — ensures predictions are consistent with episodic return progression
  - `total_loss = reward_loss + regu_lam * regu_loss`
- **Data Generation**: `delayed_reward_data()` with `start_return`/`end_return` labels
  - Each example tracks cumulative return before and after the window
  - Windows stop at episode boundaries

### Data Processing & Utilities

- `dataproc.py`: `collection_traj_data()` collects trajectories with episode boundary handling
- `mathutils.py`: Base conversion, feature hashing, Poisson confidence intervals
- `logger.py`: `ExperimentLogger` context manager for experiment tracking
- `ray_utils.py`: Ray distributed computing utilities

## Running Experiments

### Local Execution
```sh
sbin/local/rest-o1.sh  # Run O1 via Ray locally
sbin/local/rest-o2.sh  # Run O2 via Ray locally
```

### Remote/Cluster
```sh
# Cluster management (sbin/rconfig/)
cluster-spin-up.sh / cluster-shutdown.sh
cluster-state.sh / cluster-rescale.sh

# Submit jobs (sbin/rjobs/)
rjobs/rest-o1.sh  # Submit O1 to Ray cluster
```

### Direct Python
```sh
# O1 Window-Based
python src/drmdp/dfdrl/est_o1.py --env MountainCarContinuous-v0 --delay 3 --num-steps 10000

# O2 Return-Grounded
python src/drmdp/dfdrl/est_o2.py --env MountainCarContinuous-v0 \
    --delay 3 --num-steps 10000 --regu-lam 0.5
```

## Critical Architectural Patterns

### Episode Boundary Handling

Delayed reward windows MUST NEVER span episode boundaries.

When processing windows:
- Cumulative returns reset at episode starts (no carryover)
- Window generation stops at episode termination
- No data leakage across episodes

**Validation**: See `tests/drmdp/dfdrl/test_est_o2.py::test_episode_boundary_no_span`

### Delayed Reward Processing Workflow

1. **Data Collection**: Wrap env with `DelayedRewardWrapper`
   - Aggregates rewards over windows
   - Emits aggregate at window end, `None` for intermediate steps

2. **Window Generation**: `delayed_reward_data()` or `delayed_reward_data_consecutive_windows()`
   - Creates (state, action, term) sequences with aggregate reward labels
   - Maintains episode boundaries, tracks cumulative returns

3. **Training**: Model predicts per-step rewards summing to aggregate
   - O1: Direct window supervision
   - O2: Stage 1 return prediction → Stage 2 reward prediction with regularization

## Coding Style

Follow **Google's Python Style Guide** with these specific conventions:

**Code Organization (Newspaper Style)**: Organize code top-down with classes before functions, main routines before subroutines
```python
# Good - Classes first, then functions, main before helpers
class RNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        ...

    def forward(self, state, action):
        # Main routine
        out = self._preprocess(state, action)
        return self._predict(out)

    def _preprocess(self, state, action):
        # Helper/subroutine
        ...

    def _predict(self, features):
        # Helper/subroutine
        ...

def train_model(model, data):
    # Main function
    for batch in data:
        loss = _compute_loss(model, batch)
        _update_weights(model, loss)

def _compute_loss(model, batch):
    # Helper function
    ...

def _update_weights(model, loss):
    # Helper function
    ...
```

**Comments**: Keep comments focused on explaining why, not what; avoid meta-commentary
```python
# Good
# Use polynomial features to capture non-linear reward patterns
out = torch.pow(torch.unsqueeze(out, -1), self.powers)

# Avoid - don't add meta-commentary about criticality or implementation changes
# CRITICAL: now we use variable length sequences so we need padding
# IMPORTANT: this was changed from fixed to dynamic
```

**Imports**: Only import modules and types, not functions or variables directly
```python
# Good
import numpy as np
from typing import List, Dict

# Avoid
from module import function_name
```

**Variable Naming**: Use descriptive names, never single letters
```python
# Good
for idx in range(len(items)):
    for jdx in range(len(other_items)):
        ...

# Avoid
for i in range(len(items)):
    for j in range(len(other_items)):
        ...
```

**Type Annotations**: Always use typing module types with full parameterization; prefer immutable types
```python
# Good - typing module types, immutable by default
from typing import Mapping, Sequence, Tuple, Optional
def process_config(config: Mapping[str, Any]) -> Tuple[int, int]:
    ...

# Use mutable types only when mutation is needed
from typing import Dict, List
def accumulate_results(buffer: List[float]) -> Dict[str, float]:
    buffer.append(new_value)  # Mutation required
    ...

# Avoid - built-in types without parameterization
def get_bounds() -> tuple:
    ...
```

## Testing Guidelines

**Philosophy**: pytest with 80% coverage target. Tests grouped into classes by component.

### Test Organization

**Structure**: Tests mirror source code structure. For each module, create a corresponding test module:
- `src/drmdp/mathutils.py` → `tests/drmdp/test_mathutils.py`
- `src/drmdp/dfdrl/est_o1.py` → `tests/drmdp/dfdrl/test_est_o1.py`

**Grouping**: Group tests by the function/class being tested:
```python
class TestHashtrick:  # One class per function/class under test
    def test_empty_input(self):
        ...
    def test_collision_behavior(self):
        ...
```

**Naming**: `test_<specific_behavior>()`, descriptive docstrings for critical tests

### Test Categories

**Unit Tests** (80% coverage):
- Individual functions: all code paths, edge cases, error conditions
- Use `pytest.raises()` for error testing
- Fixtures for shared setup: `@pytest.fixture`
- Parameterized tests: `@pytest.mark.parametrize()`
- Numerical precision: `np.testing.assert_allclose(result, expected, atol=1e-6)`

**Integration Tests** (extensive):
- End-to-end workflows, multi-component interactions
- Episode boundary handling (critical!)
- Data consistency across transformations
- Model training loops

### Testing Special Cases

**PyTorch/TensorFlow**: Set seeds for reproducibility (`torch.manual_seed(42)`), use `model.eval()` and `torch.no_grad()` for deterministic tests

**Stochastic Components**: Verify same seed → same results, different seeds → different results

**RL Components**: Verify trajectory structure `(s, a, s', r, done)`, episode boundaries, reward consistency

### Coverage
```sh
pytest --cov=drmdp --cov-report=html
# View: htmlcov/index.html
```

## Configuration

**pytest** (`pyproject.toml`):
```toml
addopts = "-v --cov=drmdp --cov-report=xml:cobertura/coverage.xml --cov-report=term-missing"
testpaths = ["tests"]
```

**tox** (`tox.ini`): Environments for test, check-formatting, check-lint, check-lint-types. Python 3.11/3.12, uv for dependency management.

**Dependencies**: torch, tensorflow, stable-baselines3, gymnasium, ray[data,tune], cvxpy, scipy, pytest, ruff, mypy

## Project Structure

```
src/drmdp/
├── core.py, rewdelay.py, optsol.py  # Core abstractions
├── dataproc.py, mathutils.py        # Data & utilities
├── logger.py, metrics.py, ray_utils.py
└── dfdrl/
    ├── est_o0.py, est_o1.py, est_o2.py       # Reward estimation
    └── eval_est_o0.py, eval_est_o1.py, eval_est_o2.py

tests/drmdp/  # Mirrors src/ structure
sbin/         # Experiment scripts (local/, rconfig/, rjobs/)
```
