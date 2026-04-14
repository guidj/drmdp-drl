import json
import pathlib

import pytest
import torch

from drmdp.dfdrl import est_o1, eval_est_o1


def _write_config(
    directory: pathlib.Path, state_dim: int = 2, action_dim: int = 1
) -> None:
    config = {
        "spec": "o1",
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
    num_predictions: int = 3,
    state_dim: int = 2,
    action_dim: int = 1,
    model_type: str = "mlp",
) -> None:
    import numpy as np

    rng = np.random.default_rng(seed=0)
    predictions = [
        {
            "actual_reward": float(rng.standard_normal()),
            "predicted_reward": float(rng.standard_normal()),
            "per_step_predictions": rng.standard_normal(3).tolist(),
        }
        for _ in range(num_predictions)
    ]
    data = {
        "model_type": model_type,
        "final_mse": 0.5,
        "num_predictions": num_predictions,
        "predictions": predictions,
    }
    fname = f"predictions_{model_type}.json"
    with open(str(directory / fname), "w", encoding="UTF-8") as writable:
        json.dump(data, writable)


class TestLoadConfig:
    def test_returns_config_dict(self, tmp_path):
        _write_config(tmp_path, state_dim=4, action_dim=2)
        config = eval_est_o1.load_config(str(tmp_path))
        assert config["spec"] == "o1"
        assert config["state_dim"] == 4
        assert config["action_dim"] == 2

    def test_raises_file_not_found_if_config_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o1.load_config(str(tmp_path))


class TestEvaluateFromPredictionsFile:
    def test_runs_without_error(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=4)
        eval_est_o1.evaluate_from_predictions_file(
            str(tmp_path / "predictions_mlp.json"), num_examples=3
        )
        captured = capsys.readouterr()
        assert "O1" in captured.out or "Predicted" in captured.out

    def test_raises_file_not_found_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o1.evaluate_from_predictions_file(str(tmp_path / "missing.json"))

    def test_displays_fewer_than_requested_examples(self, tmp_path, capsys):
        _write_predictions(tmp_path, num_predictions=2)
        eval_est_o1.evaluate_from_predictions_file(
            str(tmp_path / "predictions_mlp.json"), num_examples=10
        )
        captured = capsys.readouterr()
        assert "2/" in captured.out


class TestLoadModel:
    def test_loads_mlp_model_in_eval_mode(self, tmp_path):
        state_dim, action_dim = 2, 1
        model = est_o1.RNetwork(state_dim=state_dim, action_dim=action_dim)
        model_path = tmp_path / "model_mlp.pt"
        torch.save(model.state_dict(), str(model_path))

        loaded = eval_est_o1.load_model(
            str(model_path),
            model_type="mlp",
            state_dim=state_dim,
            action_dim=action_dim,
        )
        assert not loaded.training

    def test_unknown_model_type_raises_value_error(self, tmp_path):
        model_path = tmp_path / "model.pt"
        model_path.touch()
        with pytest.raises(ValueError, match="Unknown model_type"):
            eval_est_o1.load_model(
                str(model_path), model_type="rnn", state_dim=2, action_dim=1
            )

    def test_raises_file_not_found_if_model_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_est_o1.load_model(
                str(tmp_path / "no_model.pt"),
                model_type="mlp",
                state_dim=2,
                action_dim=1,
            )
