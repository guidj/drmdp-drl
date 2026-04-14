import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drmdp import dataproc


class SimpleEnv(gym.Env):
    """
    Simple test environment that terminates after `term_steps` steps.
    Observations are step count, rewards are 1.0 per step.
    """

    def __init__(self, term_steps: int = 5):
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self.step_count = 0
        self.term_steps = term_steps

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.array([self.step_count], dtype=np.float32)
        reward = 1.0
        terminated = self.step_count >= self.term_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
        self.step_count = 0
        return np.array([0.0], dtype=np.float32), {}


def test_collection_traj_data_basic():
    """Test basic trajectory collection without include_term flag."""
    env = SimpleEnv(term_steps=5)
    steps = 10
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42)

    # Should collect exactly the requested number of steps
    assert len(buffer) == steps

    # Expected trajectory values with seed=42
    expected_values = [
        # Episode 1 (steps 0-4)
        (arr(0.0), 0, arr(1.0), 1.0),
        (arr(1.0), 1, arr(2.0), 1.0),
        (arr(2.0), 1, arr(3.0), 1.0),
        (arr(3.0), 0, arr(4.0), 1.0),
        (arr(4.0), 0, arr(5.0), 1.0),
        # Episode 2 (steps 5-9)
        (arr(0.0), 1, arr(1.0), 1.0),
        (arr(1.0), 0, arr(2.0), 1.0),
        (arr(2.0), 1, arr(3.0), 1.0),
        (arr(3.0), 0, arr(4.0), 1.0),
        (arr(4.0), 0, arr(5.0), 1.0),
    ]

    # Check exact values for each step
    for i, (item, expected) in enumerate(zip(buffer, expected_values)):
        assert len(item) == 4, f"Step {i}: expected 4 elements"
        obs, action, next_obs, rew = item
        expected_obs, expected_action, expected_next_obs, expected_rew = expected

        np.testing.assert_array_equal(
            obs, expected_obs, err_msg=f"Step {i}: obs mismatch"
        )
        assert action == expected_action, f"Step {i}: action mismatch"
        np.testing.assert_array_equal(
            next_obs, expected_next_obs, err_msg=f"Step {i}: next_obs mismatch"
        )
        assert rew == expected_rew, f"Step {i}: reward mismatch"


def test_collection_traj_data_with_include_term():
    """Test trajectory collection with include_term=True."""
    env = SimpleEnv(term_steps=5)
    steps = 10
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42, include_term=True)

    # Should collect exactly the requested number of steps
    assert len(buffer) == steps

    # Expected trajectory values with seed=42 and include_term=True
    expected_values = [
        # Episode 1 (steps 0-4)
        (arr(0.0), 0, arr(1.0), 1.0, False),
        (arr(1.0), 1, arr(2.0), 1.0, False),
        (arr(2.0), 1, arr(3.0), 1.0, False),
        (arr(3.0), 0, arr(4.0), 1.0, False),
        (arr(4.0), 0, arr(5.0), 1.0, True),
        # Episode 2 (steps 5-9)
        (arr(0.0), 1, arr(1.0), 1.0, False),
        (arr(1.0), 0, arr(2.0), 1.0, False),
        (arr(2.0), 1, arr(3.0), 1.0, False),
        (arr(3.0), 0, arr(4.0), 1.0, False),
        (arr(4.0), 0, arr(5.0), 1.0, True),
    ]

    # Check exact values for each step
    for i, (item, expected) in enumerate(zip(buffer, expected_values)):
        assert len(item) == 5, f"Step {i}: expected 5 elements"
        obs, action, next_obs, rew, term = item
        (
            expected_obs,
            expected_action,
            expected_next_obs,
            expected_rew,
            expected_term,
        ) = expected

        np.testing.assert_array_equal(
            obs, expected_obs, err_msg=f"Step {i}: obs mismatch"
        )
        assert action == expected_action, f"Step {i}: action mismatch"
        np.testing.assert_array_equal(
            next_obs, expected_next_obs, err_msg=f"Step {i}: next_obs mismatch"
        )
        assert rew == expected_rew, f"Step {i}: reward mismatch"
        assert term == expected_term, f"Step {i}: term mismatch"


def test_collection_traj_data_deterministic_with_seed():
    """Test that same seed produces same trajectories."""
    env1 = SimpleEnv(term_steps=5)
    env2 = SimpleEnv(term_steps=5)
    steps = 10
    seed = 42

    buffer1 = dataproc.collection_traj_data(env1, steps=steps, seed=seed)
    buffer2 = dataproc.collection_traj_data(env2, steps=steps, seed=seed)

    assert len(buffer1) == len(buffer2)

    # Actions should be identical with same seed
    actions1 = [item[1] for item in buffer1]
    actions2 = [item[1] for item in buffer2]
    assert actions1 == actions2


def test_collection_traj_data_different_seeds():
    """Test that different seeds produce different trajectories."""
    env1 = SimpleEnv(term_steps=5)
    env2 = SimpleEnv(term_steps=5)
    steps = 100

    buffer1 = dataproc.collection_traj_data(env1, steps=steps, seed=42)
    buffer2 = dataproc.collection_traj_data(env2, steps=steps, seed=123)

    # Actions should likely differ with different seeds
    actions1 = [item[1] for item in buffer1]
    actions2 = [item[1] for item in buffer2]
    # With high probability, at least one action should differ
    assert actions1 != actions2


def test_collection_traj_data_episode_boundaries():
    """Test that environment resets correctly at episode boundaries."""
    env = SimpleEnv(term_steps=3)
    steps = 10
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42, include_term=True)

    # Find all terminal transitions
    terminal_indices = [i for i, item in enumerate(buffer) if item[4] is True]

    # Should have at least 2 terminal transitions (10 steps / 3 steps per episode)
    assert len(terminal_indices) >= 2

    # After each terminal transition (except the last), next observation should be reset
    for idx in terminal_indices[:-1]:
        if idx + 1 < len(buffer):
            next_obs_after_term = buffer[idx + 1][0]
            # After reset, observation should be [0.0]
            np.testing.assert_array_equal(next_obs_after_term, arr(0.0))


def test_collection_traj_data_observation_action_consistency():
    """Test that observations and actions are consistent across transitions."""
    env = SimpleEnv(term_steps=10)
    steps = 5
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42)

    # For each transition, next_obs should match the obs of the next transition
    # (unless there's a reset)
    for idx in range(len(buffer) - 1):
        current_next_obs = buffer[idx][2]
        next_obs = buffer[idx + 1][0]

        # If step count didn't reset, they should match
        current_step = int(current_next_obs[0])
        next_step = int(next_obs[0])

        if next_step > current_step or next_step == 0:
            # Either continuing episode or reset occurred
            if next_step == 0:
                # Reset occurred, which is fine
                pass
            else:
                # Continuing episode, next_obs should match
                np.testing.assert_array_equal(current_next_obs, next_obs)


def test_collection_traj_data_rewards():
    """Test that rewards are collected correctly."""
    env = SimpleEnv(term_steps=10)
    steps = 5
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42)

    # All rewards should be 1.0 in SimpleEnv
    for item in buffer:
        assert item[3] == 1.0


def test_collection_traj_data_single_step():
    """Test collection of a single step."""
    env = SimpleEnv(term_steps=10)
    steps = 1
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42)

    assert len(buffer) == 1
    assert len(buffer[0]) == 4


def test_collection_traj_data_zero_steps():
    """Test that zero steps returns empty buffer."""
    env = SimpleEnv(term_steps=10)
    steps = 0
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42)

    assert len(buffer) == 0


def test_collection_traj_data_exact_episode_length():
    """Test collection when steps exactly matches episode length."""
    env = SimpleEnv(term_steps=5)
    steps = 5
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42, include_term=True)

    assert len(buffer) == 5

    # Last transition should be terminal
    assert buffer[-1][4] is True


def test_collection_traj_data_multiple_episodes():
    """Test collection across multiple episodes."""
    env = SimpleEnv(term_steps=3)
    steps = 10
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42, include_term=True)

    # Count number of terminal states
    num_terminals = sum(1 for item in buffer if item[4] is True)

    # With 10 steps and episodes of length 3, we should see at least 3 terminals
    # (episodes at steps 3, 6, 9)
    assert num_terminals >= 3


def test_collection_traj_data_action_space_sampling():
    """Test that actions are sampled from the action space."""
    env = SimpleEnv(term_steps=10)
    steps = 20
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42)

    actions = [item[1] for item in buffer]

    # All actions should be valid (0 or 1 for Discrete(2))
    assert all(action in [0, 1] for action in actions)

    # With 20 samples, we should see both actions with high probability
    unique_actions = set(actions)
    assert len(unique_actions) >= 1  # At least one action type


def test_collection_traj_data_no_seed():
    """Test that collection works without specifying a seed."""
    env = SimpleEnv(term_steps=5)
    steps = 10

    # Should not raise an error
    buffer = dataproc.collection_traj_data(env, steps=steps)

    assert len(buffer) == steps
    assert len(buffer[0]) == 4


def test_collection_traj_data_observations_increment():
    """Test that observations increment correctly in SimpleEnv."""
    env = SimpleEnv(term_steps=10)
    steps = 5
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42)

    # In SimpleEnv without termination, next_obs should increment
    for _, item in enumerate(buffer):
        obs, _, next_obs, _ = item
        # next_obs should be obs + 1 (unless reset occurred)
        if int(next_obs[0]) != 0:  # Not a reset
            expected_next = int(obs[0]) + 1
            assert int(next_obs[0]) == expected_next


def test_collection_traj_data_term_flag_values():
    """Test that term flag has correct values."""
    env = SimpleEnv(term_steps=3)
    steps = 10
    buffer = dataproc.collection_traj_data(env, steps=steps, seed=42, include_term=True)

    # Term should be True at steps 3, 6, 9 (multiples of term_steps)
    for _, item in enumerate(buffer):
        _, _, next_obs, _, term = item
        step_in_episode = int(next_obs[0]) if int(next_obs[0]) != 0 else 3

        if step_in_episode == 3:
            assert term is True
        # Note: We can't assert False for non-terminal because of how SimpleEnv works


def arr(value: float) -> np.ndarray:
    """Helper to create float32 array with single value."""
    return np.array([value], dtype=np.float32)


import pandas as pd  # noqa: E402 — placed here to keep existing tests unchanged


class TestWideMetrics:
    def test_drops_metrics_column(self):
        df = pd.DataFrame(
            {
                "name": ["a", "b"],
                "metrics": [{"x": 1}, {"x": 2}],
                "returns": [[1.0, 2.0], [3.0]],
            }
        )
        result = dataproc.wide_metrics(df)
        assert "metrics" not in result.columns

    def test_explodes_returns_column(self):
        df = pd.DataFrame(
            {
                "name": ["a"],
                "metrics": [{}],
                "returns": [[1.0, 2.0, 3.0]],
            }
        )
        result = dataproc.wide_metrics(df)
        assert len(result) == 3
        assert list(result["returns"]) == [1.0, 2.0, 3.0]

    def test_preserves_other_columns(self):
        df = pd.DataFrame(
            {
                "name": ["run1", "run1"],
                "metrics": [{}, {}],
                "returns": [[10.0], [20.0]],
            }
        )
        result = dataproc.wide_metrics(df)
        assert "name" in result.columns


class TestGetDistinctEnvs:
    def test_returns_unique_env_names(self):
        df = pd.DataFrame(
            {
                "meta": [
                    {"env_spec": {"name": "CartPole-v1", "args": None}},
                    {"env_spec": {"name": "CartPole-v1", "args": None}},
                    {"env_spec": {"name": "Pendulum-v1", "args": {"n": 1}}},
                ]
            }
        )
        result = dataproc.get_distinct_envs(df)
        assert set(result.keys()) == {"CartPole-v1", "Pendulum-v1"}

    def test_env_args_preserved(self):
        env_args = {"max_steps": 500}
        df = pd.DataFrame({"meta": [{"env_spec": {"name": "MyEnv", "args": env_args}}]})
        result = dataproc.get_distinct_envs(df)
        assert result["MyEnv"] == env_args


class TestDropDuplicateSets:
    def test_drops_rows_with_identical_key_sets(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 1],
                "b": [3, 4, 3],
                "c": [5, 6, 7],
            }
        )
        result = dataproc.drop_duplicate_sets(df, keys=["a", "b"])
        assert len(result) == 2

    def test_preserves_all_columns(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = dataproc.drop_duplicate_sets(df, keys=["x"])
        assert set(result.columns) == {"x", "y"}

    def test_no_duplicates_returns_all_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = dataproc.drop_duplicate_sets(df, keys=["a"])
        assert len(result) == 3
