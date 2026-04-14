import json
import pathlib

import numpy as np
import pytest
import torch

from drmdp.dfdrl import est_o2, eval_est_o2


def _write_config(
    directory: pathlib.Path, state_dim: int = 2, action_dim: int = 1
) -> None:
    config = {
        "spec": "o2",
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
            # eval_est_o2 uses len(pred["state"]) as window length
            "state": rng.standard_normal((window_len, state_dim)).tolist(),
            "actual_reward": float(rng.standard_normal()),
            "predicted_reward": float(rng.standard_normal()),
            "per_step_predictions": rng.standard_normal(window_len).tolist(),
        }
        for _ in range(num_predictions)
    ]
    data = {
        "model_type": "mlp",
        "final_mse": 0.5,  # float, not dict
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
        config = eval_est_o2.load_config(str(tmp_path))
        assert config["spec"] == "o2"
        assert config["state_dim"] == 3

    def test_raises_file_not_found_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o2.load_config(str(tmp_path))


class TestEvaluateFromPredictionsFile:
    def test_runs_without_error(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=3)
        eval_est_o2.evaluate_from_predictions_file(
            str(tmp_path / "predictions_mlp.json"), num_examples=2
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_raises_file_not_found_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o2.evaluate_from_predictions_file(str(tmp_path / "missing.json"))

    def test_displays_fewer_than_requested_examples(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=2)
        eval_est_o2.evaluate_from_predictions_file(
            str(tmp_path / "predictions_mlp.json"), num_examples=10
        )
        captured = capsys.readouterr()
        assert "2/" in captured.out


class TestLoadModel:
    def test_loads_model_in_eval_mode(self, tmp_path):
        state_dim, action_dim = 2, 1
        model = est_o2.RNetwork(state_dim=state_dim, action_dim=action_dim)
        model_path = tmp_path / "model_mlp.pt"
        torch.save(model.state_dict(), str(model_path))

        loaded = eval_est_o2.load_model(str(model_path), state_dim, action_dim)
        assert not loaded.training

    def test_raises_file_not_found_if_model_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o2.load_model(str(tmp_path / "no_model.pt"), 2, 1)
