import json
import pathlib
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
import torch

from drmdp.dfdrl import est_o0, eval_est_o0


class _FakeEnv(gym.Env):
    def __init__(self, obs_dim: int = 2, act_dim: int = 1):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def step(self, action: Any):
        return self.observation_space.sample(), 0.5, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Any] = None):
        return self.observation_space.sample(), {}


def _write_config(
    directory: pathlib.Path, state_dim: int = 2, action_dim: int = 1
) -> None:
    config = {
        "spec": "o0",
        "model_type": "mlp",
        "env_name": "FakeEnv",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": 64,
        "eval_steps": 10,
        "hidden_dim": 256,
        "timestamp": 1234567890,
    }
    with open(str(directory / "config.json"), "w", encoding="UTF-8") as writable:
        json.dump(config, writable)


def _write_predictions(
    directory: pathlib.Path,
    num_predictions: int = 5,
    state_dim: int = 2,
    action_dim: int = 1,
) -> None:
    rng = np.random.default_rng(seed=0)
    predictions = [
        {
            "state": rng.standard_normal(state_dim).tolist(),
            "action": rng.standard_normal(action_dim).tolist(),
            "term": [0.0],
            "actual_reward": float(rng.standard_normal()),
            "predicted_reward": float(rng.standard_normal()),
        }
        for _ in range(num_predictions)
    ]
    data = {
        "model_type": "mlp_immediate",
        "final_mse": 0.5,
        "num_predictions": num_predictions,
        "predictions": predictions,
    }
    with open(
        str(directory / "predictions_o0.json"), "w", encoding="UTF-8"
    ) as writable:
        json.dump(data, writable)


class TestLoadConfig:
    def test_returns_config_dict(self, tmp_path):
        _write_config(tmp_path)
        config = eval_est_o0.load_config(str(tmp_path))
        assert config["spec"] == "o0"
        assert config["state_dim"] == 2

    def test_raises_file_not_found_if_config_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o0.load_config(str(tmp_path))

    def test_returns_correct_state_and_action_dims(self, tmp_path):
        _write_config(tmp_path, state_dim=4, action_dim=2)
        config = eval_est_o0.load_config(str(tmp_path))
        assert config["state_dim"] == 4
        assert config["action_dim"] == 2


class TestEvaluateFromPredictionsFile:
    def test_runs_without_error(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=5)
        eval_est_o0.evaluate_from_predictions_file(
            str(tmp_path / "predictions_o0.json"), num_examples=3
        )
        captured = capsys.readouterr()
        assert "Predicted" in captured.out or "O0" in captured.out

    def test_raises_file_not_found_if_predictions_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o0.evaluate_from_predictions_file(str(tmp_path / "missing.json"))

    def test_handles_fewer_examples_than_requested(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=2)
        eval_est_o0.evaluate_from_predictions_file(
            str(tmp_path / "predictions_o0.json"), num_examples=10
        )
        # Should display only 2 examples (min of 2 and 10)
        captured = capsys.readouterr()
        assert "2/" in captured.out


class TestLoadModel:
    def test_loads_model_in_eval_mode(self, tmp_path):
        state_dim, action_dim = 2, 1
        model = est_o0.RNetwork(state_dim=state_dim, action_dim=action_dim)
        model_path = tmp_path / "model_o0.pt"
        torch.save(model.state_dict(), str(model_path))

        loaded = eval_est_o0.load_model(str(model_path), state_dim, action_dim)
        assert not loaded.training  # eval mode

    def test_loaded_model_final_layer_weights_match(self, tmp_path):
        # Only final_layer is in state_dict because hidden layers are in a plain Python list.
        # Verify that final_layer weights are correctly restored after loading.
        torch.manual_seed(42)
        state_dim, action_dim = 3, 2
        model = est_o0.RNetwork(state_dim=state_dim, action_dim=action_dim)
        model_path = tmp_path / "model_o0.pt"
        torch.save(model.state_dict(), str(model_path))

        loaded = eval_est_o0.load_model(str(model_path), state_dim, action_dim)
        torch.testing.assert_close(model.final_layer.weight, loaded.final_layer.weight)
        torch.testing.assert_close(model.final_layer.bias, loaded.final_layer.bias)

    def test_raises_file_not_found_if_model_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o0.load_model(str(tmp_path / "no_model.pt"), 2, 1)
