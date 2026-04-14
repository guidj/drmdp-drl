import json
import pathlib
from typing import Any, Optional
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from drmdp import rewdelay
from drmdp.dfdrl import est_o3, eval_est_o3


class _TermEnv(gym.Env):
    """Env that terminates after a fixed number of steps."""

    def __init__(self, obs_dim: int = 2, act_dim: int = 1, steps_to_term: int = 4):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        self._steps = 0
        self._steps_to_term = steps_to_term

    def step(self, action: Any):
        self._steps += 1
        obs = self.observation_space.sample()
        terminated = self._steps >= self._steps_to_term
        return obs, 1.0, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Any] = None):
        self._steps = 0
        return self.observation_space.sample(), {}


def _write_config(
    directory: pathlib.Path, state_dim: int = 2, action_dim: int = 1
) -> None:
    config = {
        "spec": "o3",
        "model_type": "mlp",
        "env_name": "FakeEnv",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": 64,
        "hidden_dim": 256,
        "timestamp": 1234567890,
    }
    with open(str(directory / "config.json"), "w", encoding="UTF-8") as writable:
        json.dump(config, writable)


def _write_predictions(
    directory: pathlib.Path,
    num_predictions: int = 3,
    state_dim: int = 2,
    window_len: int = 3,
) -> None:
    rng = np.random.default_rng(seed=0)
    predictions = [
        {
            # eval_est_o3 uses len(pred["state"]) as window length
            "state": rng.standard_normal((window_len, state_dim)).tolist(),
            "actual_reward": float(rng.standard_normal()),
            "predicted_reward": float(rng.standard_normal()),
            "per_step_predictions": rng.standard_normal(window_len).tolist(),
        }
        for _ in range(num_predictions)
    ]
    data = {
        "model_type": "mlp",
        "final_mse": {"reward": 0.3, "regu": 0.1, "total": 0.4},  # dict, not float
        "num_predictions": num_predictions,
        "predictions": predictions,
    }
    with open(
        str(directory / "predictions_mlp.json"), "w", encoding="UTF-8"
    ) as writable:
        json.dump(data, writable)


class TestLoadConfig:
    def test_returns_config_dict(self, tmp_path):
        _write_config(tmp_path, state_dim=3, action_dim=2)
        config = eval_est_o3.load_config(str(tmp_path))
        assert config["spec"] == "o3"
        assert config["state_dim"] == 3

    def test_raises_file_not_found_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o3.load_config(str(tmp_path))


class TestEvaluateFromPredictionsFile:
    def test_runs_without_error(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=3)
        eval_est_o3.evaluate_from_predictions_file(
            str(tmp_path / "predictions_mlp.json"), num_examples=2
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_raises_file_not_found_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o3.evaluate_from_predictions_file(str(tmp_path / "missing.json"))

    def test_displays_fewer_than_requested_examples(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=2)
        eval_est_o3.evaluate_from_predictions_file(
            str(tmp_path / "predictions_mlp.json"), num_examples=10
        )
        captured = capsys.readouterr()
        assert "2/" in captured.out

    def test_dict_mse_printed_per_component(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=2)
        eval_est_o3.evaluate_from_predictions_file(
            str(tmp_path / "predictions_mlp.json"), num_examples=2
        )
        captured = capsys.readouterr()
        assert "MSE" in captured.out
        assert "total" in captured.out


class TestLoadModel:
    def test_loads_model_in_eval_mode(self, tmp_path):
        state_dim, action_dim = 2, 1
        model = est_o3.RNetwork(state_dim=state_dim, action_dim=action_dim)
        model_path = tmp_path / "model_mlp.pt"
        torch.save(model.state_dict(), str(model_path))

        loaded = eval_est_o3.load_model(str(model_path), state_dim, action_dim)
        assert not loaded.training

    def test_raises_file_not_found_if_model_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o3.load_model(str(tmp_path / "no_model.pt"), 2, 1)


class TestEvaluateInteractive:
    def test_runs_one_episode_with_full_windows(self, capsys):
        # FixedDelay(2) + 4-step env → 2 complete windows
        model = est_o3.RNetwork(state_dim=2, action_dim=1)
        model.eval()
        env = _TermEnv(obs_dim=2, act_dim=1, steps_to_term=4)
        delay = rewdelay.FixedDelay(2)
        eval_est_o3.evaluate_interactive(model, env, delay, num_episodes=1)
        captured = capsys.readouterr()
        assert "MAE" in captured.out or "RMSE" in captured.out

    def test_no_complete_windows_does_not_crash(self, capsys):
        # Env terminates in 1 step, delay=3 → no complete windows
        model = est_o3.RNetwork(state_dim=2, action_dim=1)
        model.eval()
        env = _TermEnv(obs_dim=2, act_dim=1, steps_to_term=1)
        delay = rewdelay.FixedDelay(3)
        eval_est_o3.evaluate_interactive(model, env, delay, num_episodes=1)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_runs_overall_stats(self, capsys):
        model = est_o3.RNetwork(state_dim=2, action_dim=1)
        model.eval()
        env = _TermEnv(obs_dim=2, act_dim=1, steps_to_term=4)
        delay = rewdelay.FixedDelay(2)
        eval_est_o3.evaluate_interactive(model, env, delay, num_episodes=2)
        captured = capsys.readouterr()
        assert "Overall Reward MAE" in captured.out


class TestMain:
    def test_predictions_mode(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=3)
        _write_config(tmp_path)
        with patch(
            "sys.argv",
            [
                "prog",
                "--model-dir",
                str(tmp_path),
                "--mode",
                "predictions",
                "--num-examples",
                "2",
            ],
        ):
            eval_est_o3.main()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_interactive_mode(self, tmp_path, capsys):
        state_dim, action_dim = 2, 1
        _write_config(tmp_path, state_dim=state_dim, action_dim=action_dim)
        model = est_o3.RNetwork(state_dim=state_dim, action_dim=action_dim)
        torch.save(model.state_dict(), str(tmp_path / "model_mlp.pt"))
        env = _TermEnv(obs_dim=state_dim, act_dim=action_dim, steps_to_term=4)
        with patch(
            "sys.argv",
            [
                "prog",
                "--model-dir",
                str(tmp_path),
                "--mode",
                "interactive",
                "--num-episodes",
                "1",
                "--delay",
                "2",
            ],
        ):
            with patch("gymnasium.make", return_value=env):
                eval_est_o3.main()
        captured = capsys.readouterr()
        assert "MAE" in captured.out or "RMSE" in captured.out
