"""
Tests for drmdp.control.runner: RewardModelUpdateCallback and run().
"""

import dataclasses
import os
import tempfile
from typing import Any, Mapping, Optional, Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from drmdp.control import base, runner

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _TrackingRewardModel(base.RewardModel):
    """Reward model that tracks update calls and returns a fixed constant."""

    def __init__(self, constant: float = 0.0):
        self._constant = constant
        self.update_calls: int = 0
        self.trajectories_received: list = []

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        return np.full(len(observations), self._constant, dtype=np.float32)

    def update(self, trajectories: Sequence[base.Trajectory]) -> Mapping[str, float]:
        self.update_calls += 1
        self.trajectories_received.extend(trajectories)
        return {"buffer_size": float(len(self.trajectories_received))}


class _MockReplayBuffer:
    """Minimal replay buffer stub that records reset() calls."""

    def __init__(self):
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _MockLogger:
    """Minimal ExperimentLogger stub."""

    def __init__(self):
        self.log_calls: list = []

    def log(self, episode, steps, returns, info=None):
        self.log_calls.append(
            {"episode": episode, "steps": steps, "returns": returns, "info": info}
        )


def _build_callback(
    reward_model: Optional[base.RewardModel] = None,
    update_every: int = 100,
    clear_buffer: bool = False,
    log_freq: int = 1,
    exp_logger: Any = None,
) -> runner.RewardModelUpdateCallback:
    """Construct a callback with sensible defaults for unit testing."""
    if reward_model is None:
        reward_model = _TrackingRewardModel()
    if exp_logger is None:
        exp_logger = _MockLogger()
    return runner.RewardModelUpdateCallback(
        reward_model=reward_model,
        update_every_n_steps=update_every,
        clear_buffer_on_update=clear_buffer,
        log_episode_frequency=log_freq,
        exp_logger=exp_logger,
    )


def _make_mock_sac(obs_dim: int = 3, act_dim: int = 1) -> MagicMock:
    """Return a minimal SAC mock with _last_obs and replay_buffer."""
    sac = MagicMock()
    sac._last_obs = np.zeros((1, obs_dim), dtype=np.float32)
    sac.replay_buffer = _MockReplayBuffer()
    return sac


def _step_callback(
    callback: runner.RewardModelUpdateCallback,
    sac_model: Any,
    obs_before: np.ndarray,
    action: np.ndarray,
    reward: float,
    done: bool,
    num_timesteps: int,
) -> bool:
    """Simulate one SB3 callback step by setting required attributes/locals."""
    sac_model._last_obs = obs_before[np.newaxis]  # (1, obs_dim)
    callback.model = sac_model
    callback.locals = {
        "actions": action[np.newaxis],  # (1, act_dim)
        "rewards": np.array([reward]),
        "dones": np.array([done]),
        "infos": [{}],
    }
    callback.num_timesteps = num_timesteps
    return callback._on_step()


# ---------------------------------------------------------------------------
# TestRewardModelUpdateCallback
# ---------------------------------------------------------------------------


class TestRewardModelUpdateCallback:
    def test_trajectory_built_correctly_across_episode(self):
        """Completed Trajectory has correct observations, actions, episode_return."""
        model = _TrackingRewardModel()
        callback = _build_callback(reward_model=model, update_every=1000)
        sac = _make_mock_sac(obs_dim=3, act_dim=1)

        rng = np.random.default_rng(0)
        obs_seq = [rng.uniform(size=3).astype(np.float32) for _ in range(4)]
        actions = [rng.uniform(size=1).astype(np.float32) for _ in range(4)]
        rewards = [1.0, 2.0, 3.0, 4.0]

        for step_idx in range(4):
            done = step_idx == 3
            _step_callback(
                callback,
                sac,
                obs_before=obs_seq[step_idx],
                action=actions[step_idx],
                reward=rewards[step_idx],
                done=done,
                num_timesteps=step_idx + 1,
            )

        assert len(callback._pending_trajectories) == 1
        traj = callback._pending_trajectories[0]
        np.testing.assert_allclose(traj.observations, np.stack(obs_seq), atol=1e-6)
        np.testing.assert_allclose(traj.actions, np.stack(actions), atol=1e-6)
        assert traj.episode_return == pytest.approx(sum(rewards))

    def test_update_called_at_configured_interval(self):
        """reward_model.update() is called when num_timesteps is a multiple of update_every."""
        model = _TrackingRewardModel()
        update_every = 5
        callback = _build_callback(reward_model=model, update_every=update_every)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        # Simulate one episode ending at step 3, then steps up to 5.
        for step_idx in range(1, 6):
            done = step_idx == 3
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert model.update_calls == 1

    def test_update_not_called_before_interval(self):
        """No update fires if num_timesteps has not reached the interval yet."""
        model = _TrackingRewardModel()
        callback = _build_callback(reward_model=model, update_every=100)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        # Run 4 steps with episode end at step 2.
        for step_idx in range(1, 5):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert model.update_calls == 0

    def test_clear_buffer_on_update(self):
        """replay_buffer.reset() is called after the reward model update when flag is set."""
        model = _TrackingRewardModel()
        callback = _build_callback(
            reward_model=model, update_every=3, clear_buffer=True
        )
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        # Episode ends at step 2; interval fires at step 3.
        for step_idx in range(1, 4):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert sac.replay_buffer.reset_count == 1

    def test_clear_buffer_not_called_when_flag_off(self):
        """replay_buffer.reset() is NOT called when clear_buffer_on_update=False."""
        model = _TrackingRewardModel()
        callback = _build_callback(
            reward_model=model, update_every=3, clear_buffer=False
        )
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for step_idx in range(1, 4):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert sac.replay_buffer.reset_count == 0

    def test_logger_called_at_log_frequency(self):
        """ExperimentLogger.log() is called every log_episode_frequency episodes."""
        exp_logger = _MockLogger()
        callback = _build_callback(log_freq=2, exp_logger=exp_logger)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        # Run 4 episodes, each 3 steps long.
        timestep = 0
        for episode_idx in range(4):
            for step_idx in range(3):
                timestep += 1
                done = step_idx == 2
                _step_callback(callback, sac, obs, action, 1.0, done, timestep)

        # Should log at episodes 2 and 4.
        assert len(exp_logger.log_calls) == 2

    def test_training_end_flushes_pending_trajectories(self):
        """_on_training_end() passes remaining pending trajectories to the model."""
        model = _TrackingRewardModel()
        callback = _build_callback(reward_model=model, update_every=1000)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        # One episode ends (step 2 done), but interval never fires.
        for step_idx in range(1, 4):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert model.update_calls == 0  # interval hasn't fired

        callback._on_training_end()

        assert model.update_calls == 1


# ---------------------------------------------------------------------------
# TestRun (integration)
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.fixture()
    def output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def _base_args(self, output_dir: str) -> runner.TrainingArgs:
        return runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            max_episode_steps=50,
            reward_model_type="ircr",
            update_every_n_steps=50,
            clear_buffer_on_update=False,
            reward_model_kwargs={"max_buffer_size": 20, "k_neighbors": 3},
            num_steps=200,
            sac_learning_rate=3e-4,
            sac_buffer_size=1000,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_episode_frequency=1,
            output_dir=output_dir,
            seed=42,
        )

    def test_run_completes_without_error(self, output_dir):
        """run() exits without raising for a short Pendulum-v1 experiment."""
        args = self._base_args(output_dir)
        runner.run(args)  # should not raise

    def test_output_files_created(self, output_dir):
        """After run(), experiment-logs.jsonl and experiment-params.json exist."""
        args = self._base_args(output_dir)
        runner.run(args)

        assert os.path.isfile(os.path.join(output_dir, "experiment-logs.jsonl"))
        assert os.path.isfile(os.path.join(output_dir, "experiment-params.json"))

    def test_sac_model_saved(self, output_dir):
        """After run(), the SAC model checkpoint file exists."""
        args = self._base_args(output_dir)
        runner.run(args)

        assert os.path.isfile(os.path.join(output_dir, "sac_model.zip"))

    def test_clear_buffer_on_update_does_not_crash(self, output_dir):
        """run() with clear_buffer_on_update=True completes without error."""
        args = dataclasses.replace(
            self._base_args(output_dir), clear_buffer_on_update=True
        )
        runner.run(args)


# ---------------------------------------------------------------------------
# TestParseRewardModelKwargs
# ---------------------------------------------------------------------------


class TestHCLoggingCallback:
    def _build_hc_callback(self, log_freq: int = 1) -> runner._HCLoggingCallback:
        exp_logger = _MockLogger()
        return runner._HCLoggingCallback(
            log_episode_frequency=log_freq,
            exp_logger=exp_logger,
        )

    def _step_hc(
        self,
        callback: runner._HCLoggingCallback,
        done: bool,
        timestep: int,
        return_val: float = 1.0,
    ) -> bool:
        callback.num_timesteps = timestep
        callback.locals = {
            "dones": np.array([done]),
            "infos": [{"true_episode_return": return_val}],
        }
        return callback._on_step()

    def test_on_step_returns_true(self):
        callback = self._build_hc_callback()
        result = self._step_hc(callback, done=False, timestep=1)
        assert result is True

    def test_episode_count_increments_on_done(self):
        callback = self._build_hc_callback()
        self._step_hc(callback, done=True, timestep=1)
        assert callback._episode_count == 1

    def test_episode_steps_reset_after_done(self):
        callback = self._build_hc_callback()
        self._step_hc(callback, done=False, timestep=1)
        self._step_hc(callback, done=True, timestep=2)
        assert callback._episode_steps == 0

    def test_logger_called_at_log_frequency(self):
        log_mock = _MockLogger()
        callback = runner._HCLoggingCallback(
            log_episode_frequency=2, exp_logger=log_mock
        )
        for timestep in range(1, 5):
            done = timestep % 2 == 0
            callback.num_timesteps = timestep
            callback.locals = {
                "dones": np.array([done]),
                "infos": [{"true_episode_return": 1.0}],
            }
            callback._on_step()
        # Episodes 1 and 2 done; log at every 2 → only episode 2 logged
        assert len(log_mock.log_calls) == 1


class TestMakeRewardModel:
    def test_ircr_type_returns_ircr_instance(self, tmp_path):
        args = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            max_episode_steps=50,
            reward_model_type="ircr",
            reward_model_kwargs={"max_buffer_size": 10, "k_neighbors": 1},
            update_every_n_steps=100,
            clear_buffer_on_update=False,
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=100,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_episode_frequency=1,
            output_dir=str(tmp_path),
            seed=None,
        )
        model = runner._make_reward_model(args)
        from drmdp.control import ircr

        assert isinstance(model, ircr.IRCRRewardModel)

    def test_unknown_type_raises_value_error(self, tmp_path):
        args = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            max_episode_steps=50,
            reward_model_type="unknown_model",
            reward_model_kwargs={},
            update_every_n_steps=100,
            clear_buffer_on_update=False,
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=100,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_episode_frequency=1,
            output_dir=str(tmp_path),
            seed=None,
        )
        with pytest.raises(ValueError):
            runner._make_reward_model(args)


class TestParseArgs:
    def test_defaults_are_valid_training_args(self, tmp_path):
        with patch("sys.argv", ["prog", "--output-dir", str(tmp_path)]):
            args = runner.parse_args()
        assert isinstance(args, runner.TrainingArgs)
        assert args.env == "MountainCarContinuous-v0"
        assert args.delay == 3

    def test_custom_env_and_delay(self, tmp_path):
        with patch(
            "sys.argv",
            [
                "prog",
                "--env",
                "Pendulum-v1",
                "--delay",
                "5",
                "--output-dir",
                str(tmp_path),
            ],
        ):
            args = runner.parse_args()
        assert args.env == "Pendulum-v1"
        assert args.delay == 5

    def test_reward_model_kwarg_parsed(self, tmp_path):
        with patch(
            "sys.argv",
            [
                "prog",
                "--reward-model-kwarg",
                "max_buffer_size=50",
                "--output-dir",
                str(tmp_path),
            ],
        ):
            args = runner.parse_args()
        assert args.reward_model_kwargs["max_buffer_size"] == 50

    def test_agent_type_hc(self, tmp_path):
        with patch(
            "sys.argv",
            ["prog", "--agent-type", "hc", "--output-dir", str(tmp_path)],
        ):
            args = runner.parse_args()
        assert args.agent_type == "hc"

    def test_clear_buffer_flag(self, tmp_path):
        with patch(
            "sys.argv",
            ["prog", "--clear-buffer-on-update", "--output-dir", str(tmp_path)],
        ):
            args = runner.parse_args()
        assert args.clear_buffer_on_update is True


class TestParseRewardModelKwargs:
    def test_empty_list_returns_empty_mapping(self):
        """An empty input list produces an empty mapping."""
        result = runner.parse_reward_model_kwargs([])
        assert result == {}

    def test_int_value_parsed(self):
        """Integer strings are coerced to int."""
        result = runner.parse_reward_model_kwargs(["max_buffer_size=200"])
        assert result["max_buffer_size"] == 200
        assert isinstance(result["max_buffer_size"], int)

    def test_float_value_parsed(self):
        """Float strings are coerced to float."""
        result = runner.parse_reward_model_kwargs(["alpha=0.5"])
        assert result["alpha"] == pytest.approx(0.5)
        assert isinstance(result["alpha"], float)

    def test_bool_value_parsed(self):
        """Boolean literals True/False are coerced to bool."""
        result = runner.parse_reward_model_kwargs(["flag=True"])
        assert result["flag"] is True

    def test_string_fallback_for_non_literal(self):
        """Values that are not valid Python literals are kept as plain strings."""
        result = runner.parse_reward_model_kwargs(["mode=cosine"])
        assert result["mode"] == "cosine"
        assert isinstance(result["mode"], str)

    def test_multiple_pairs(self):
        """Multiple key=value pairs are all parsed into the same mapping."""
        result = runner.parse_reward_model_kwargs(
            ["max_buffer_size=200", "k_neighbors=5"]
        )
        assert result == {"max_buffer_size": 200, "k_neighbors": 5}
