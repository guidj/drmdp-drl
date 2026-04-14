import json
import pathlib
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from drmdp.dfdrl import est_o0


class _FakeEnv(gym.Env):
    """Minimal gym env that satisfies est_o0.train() interface."""

    def __init__(self, obs_dim: int = 2, act_dim: int = 1):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def step(self, action: Any):
        return self.observation_space.sample(), 0.0, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Any] = None):
        return self.observation_space.sample(), {}


def _make_buffer(
    num_samples: int = 10,
    state_dim: int = 2,
    action_dim: int = 1,
) -> List[Tuple[Any, ...]]:
    """Create a synthetic (state, action, next_state, reward, term) buffer."""
    rng = np.random.default_rng(seed=0)
    buffer = []
    for idx in range(num_samples):
        state = rng.standard_normal(state_dim).astype(np.float32)
        action = rng.standard_normal(action_dim).astype(np.float32)
        next_state = rng.standard_normal(state_dim).astype(np.float32)
        reward = rng.standard_normal()
        term = idx == (num_samples - 1)
        buffer.append((state, action, next_state, reward, term))
    return buffer


def _make_dict_dataset(
    num_samples: int = 10,
    state_dim: int = 2,
    action_dim: int = 1,
) -> est_o0.DictDataset:
    """Create a synthetic DictDataset for testing evaluate_model."""
    torch.manual_seed(0)
    states = torch.randn(num_samples, state_dim)
    actions = torch.randn(num_samples, action_dim)
    terms = torch.zeros(num_samples, 1)
    rewards = torch.randn(num_samples)
    inputs = {"state": states, "action": actions, "term": terms}
    return est_o0.DictDataset(inputs, rewards)


class TestRNetwork:
    def test_forward_output_shape(self):
        torch.manual_seed(42)
        model = est_o0.RNetwork(state_dim=3, action_dim=2)
        state = torch.randn(4, 3)
        action = torch.randn(4, 2)
        term = torch.zeros(4, 1)
        output = model(state, action, term)
        assert output.shape == (4, 1)

    def test_forward_with_polynomial_features(self):
        torch.manual_seed(42)
        # powers=2 doubles the input features
        model = est_o0.RNetwork(state_dim=2, action_dim=1, powers=2)
        state = torch.randn(3, 2)
        action = torch.randn(3, 1)
        term = torch.zeros(3, 1)
        output = model(state, action, term)
        assert output.shape == (3, 1)

    def test_forward_with_zero_hidden_layers(self):
        torch.manual_seed(42)
        model = est_o0.RNetwork(state_dim=2, action_dim=1, num_hidden_layers=0)
        state = torch.randn(5, 2)
        action = torch.randn(5, 1)
        term = torch.zeros(5, 1)
        output = model(state, action, term)
        assert output.shape == (5, 1)

    def test_forward_deterministic_with_same_seed(self):
        torch.manual_seed(7)
        model_a = est_o0.RNetwork(state_dim=2, action_dim=1)
        torch.manual_seed(7)
        model_b = est_o0.RNetwork(state_dim=2, action_dim=1)
        state = torch.tensor([[0.5, -0.3]])
        action = torch.tensor([[0.1]])
        term = torch.zeros(1, 1)
        out_a = model_a(state, action, term)
        out_b = model_b(state, action, term)
        torch.testing.assert_close(out_a, out_b)

    def test_gradients_flow_through_network(self):
        torch.manual_seed(42)
        model = est_o0.RNetwork(state_dim=2, action_dim=1)
        state = torch.randn(4, 2)
        action = torch.randn(4, 1)
        term = torch.zeros(4, 1)
        output = model(state, action, term)
        loss = output.mean()
        loss.backward()
        assert model.final_layer.weight.grad is not None


class TestDictDataset:
    def test_len_matches_input(self):
        dataset = _make_dict_dataset(num_samples=7)
        assert len(dataset) == 7

    def test_getitem_returns_dict_and_label(self):
        dataset = _make_dict_dataset(num_samples=5, state_dim=3, action_dim=2)
        inputs_dict, label = dataset[0]
        assert isinstance(inputs_dict, dict)
        assert "state" in inputs_dict
        assert "action" in inputs_dict
        assert "term" in inputs_dict
        assert inputs_dict["state"].shape == (3,)
        assert inputs_dict["action"].shape == (2,)
        assert isinstance(label, torch.Tensor)

    def test_single_item_dataset(self):
        dataset = _make_dict_dataset(num_samples=1)
        assert len(dataset) == 1
        inputs_dict, label = dataset[0]
        assert label.ndim == 0  # scalar tensor


class TestImmediateRewardData:
    def test_output_length_matches_buffer_size(self):
        buffer = _make_buffer(num_samples=8)
        examples = est_o0.immediate_reward_data(buffer)
        assert len(examples) == 8

    def test_output_contains_required_keys(self):
        buffer = _make_buffer(num_samples=3)
        examples = est_o0.immediate_reward_data(buffer)
        inputs_dict, _ = examples[0]
        assert "state" in inputs_dict
        assert "action" in inputs_dict
        assert "term" in inputs_dict

    def test_tensor_dtypes_are_float32(self):
        buffer = _make_buffer(num_samples=4)
        examples = est_o0.immediate_reward_data(buffer)
        for inputs_dict, label in examples:
            assert inputs_dict["state"].dtype == torch.float32
            assert inputs_dict["action"].dtype == torch.float32
            assert inputs_dict["term"].dtype == torch.float32
            assert label.dtype == torch.float32

    def test_term_flag_converted_correctly(self):
        # Last sample has term=True, all others have term=False
        buffer = _make_buffer(num_samples=5)
        examples = est_o0.immediate_reward_data(buffer)
        # First examples should have term=0.0
        assert examples[0][0]["term"].item() == 0.0
        # Last example has term=True → 1.0
        assert examples[-1][0]["term"].item() == 1.0


class TestCreateTimestampedOutputDir:
    def test_creates_directory_under_o0_spec(self, tmp_path):
        output_dir = est_o0.create_timestamped_output_dir(str(tmp_path))
        assert output_dir.exists()
        assert output_dir.parent.name == "o0"
        assert output_dir.parent.parent == tmp_path

    def test_returns_pathlib_path(self, tmp_path):
        output_dir = est_o0.create_timestamped_output_dir(str(tmp_path))
        assert isinstance(output_dir, pathlib.Path)

    def test_directory_name_is_numeric_timestamp(self, tmp_path):
        output_dir = est_o0.create_timestamped_output_dir(str(tmp_path))
        assert output_dir.name.isdigit()


class TestEvaluateModel:
    def test_returns_mse_and_predictions_list(self):
        torch.manual_seed(42)
        model = est_o0.RNetwork(state_dim=2, action_dim=1)
        dataset = _make_dict_dataset(num_samples=10)
        mse, predictions = est_o0.evaluate_model(
            model, dataset, batch_size=5, collect_predictions=True
        )
        assert isinstance(mse, float)
        assert isinstance(predictions, list)
        assert len(predictions) == 10

    def test_collect_predictions_false_returns_empty_list(self):
        torch.manual_seed(42)
        model = est_o0.RNetwork(state_dim=2, action_dim=1)
        dataset = _make_dict_dataset(num_samples=10)
        mse, predictions = est_o0.evaluate_model(
            model, dataset, batch_size=5, collect_predictions=False
        )
        assert isinstance(mse, float)
        assert predictions == []

    def test_max_batches_limits_evaluation(self):
        torch.manual_seed(42)
        model = est_o0.RNetwork(state_dim=2, action_dim=1)
        dataset = _make_dict_dataset(num_samples=20)
        mse, predictions = est_o0.evaluate_model(
            model, dataset, batch_size=5, collect_predictions=False, max_batches=1
        )
        assert isinstance(mse, float)
        assert predictions == []

    def test_mse_is_non_negative(self):
        torch.manual_seed(42)
        model = est_o0.RNetwork(state_dim=2, action_dim=1)
        dataset = _make_dict_dataset(num_samples=8)
        mse, _ = est_o0.evaluate_model(
            model, dataset, batch_size=4, collect_predictions=False
        )
        assert mse >= 0.0


class TestTrain:
    def test_train_creates_output_files(self, tmp_path):
        """train() on a tiny dataset should produce all expected output files."""
        torch.manual_seed(0)
        env = _FakeEnv(obs_dim=2, act_dim=1)
        dataset = _make_dict_dataset(num_samples=20, state_dim=2, action_dim=1)
        est_o0.train(
            env, dataset, batch_size=10, eval_steps=1, output_dir=str(tmp_path)
        )
        spec_dirs = list((tmp_path / "o0").iterdir())
        assert len(spec_dirs) == 1
        output_dir = spec_dirs[0]
        assert (output_dir / "predictions_o0.json").exists()
        assert (output_dir / "metrics_o0.json").exists()
        assert (output_dir / "model_o0.pt").exists()
        assert (output_dir / "config.json").exists()

    def test_train_returns_mse_and_predictions(self, tmp_path):
        torch.manual_seed(1)
        env = _FakeEnv(obs_dim=2, act_dim=1)
        dataset = _make_dict_dataset(num_samples=20, state_dim=2, action_dim=1)
        final_mse, predictions = est_o0.train(
            env, dataset, batch_size=10, eval_steps=1, output_dir=str(tmp_path)
        )
        assert isinstance(final_mse, float)
        assert final_mse >= 0.0
        assert isinstance(predictions, list)
        assert len(predictions) > 0

    def test_config_json_contains_expected_keys(self, tmp_path):
        torch.manual_seed(2)
        env = _FakeEnv(obs_dim=3, act_dim=2)
        dataset = _make_dict_dataset(num_samples=20, state_dim=3, action_dim=2)
        est_o0.train(
            env, dataset, batch_size=10, eval_steps=1, output_dir=str(tmp_path)
        )
        spec_dirs = list((tmp_path / "o0").iterdir())
        config_path = spec_dirs[0] / "config.json"
        with open(str(config_path), "r", encoding="UTF-8") as readable:
            config = json.load(readable)
        assert config["state_dim"] == 3
        assert config["action_dim"] == 2
        assert config["spec"] == "o0"
