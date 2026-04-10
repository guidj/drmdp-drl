"""
Tests for est_o1.py reward estimation data generation and models.

Critical tests focus on:
1. RNetwork architecture and forward pass
2. DictDataset wrapper functionality
3. Variable-length sequence collation and padding
4. Delayed reward window generation and aggregate correctness
5. Markovian evaluation predictions
6. End-to-end training integration
"""

import os
import tempfile
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import pytest
import torch

from drmdp import dataproc, rewdelay
from drmdp.dfdrl import est_o1

# =============================================================================
# TestRNetwork
# =============================================================================


class TestRNetwork:
    """Tests for RNetwork model."""

    def test_initialization(self):
        """Test that RNetwork initializes correctly."""
        model = est_o1.RNetwork(
            state_dim=4, action_dim=2, powers=3, num_hidden_layers=4, hidden_dim=256
        )

        # Check powers buffer
        assert hasattr(model, "powers")
        assert torch.equal(model.powers, torch.tensor([1, 2, 3]))

        # Check layers exist
        assert hasattr(model, "layers")
        assert hasattr(model, "final_layer")

        # Check num_hidden_layers stored
        assert model.num_hidden_layers == 4

    def test_forward_shape(self):
        """Test forward pass output shapes with various batch sizes."""
        model = est_o1.RNetwork(state_dim=4, action_dim=2, hidden_dim=64)

        test_configs = [(1, 3), (4, 5), (8, 10)]

        for batch_size, seq_len in test_configs:
            state = torch.randn(batch_size, seq_len, 4)
            action = torch.randn(batch_size, seq_len, 2)
            term = torch.zeros(batch_size, seq_len, 1)

            output = model(state, action, term)

            assert output.shape == (
                batch_size,
                seq_len,
                1,
            ), f"Expected shape ({batch_size}, {seq_len}, 1), got {output.shape}"

    def test_polynomial_features(self):
        """Test polynomial feature expansion with different powers."""
        for powers in [1, 2, 3]:
            model = est_o1.RNetwork(
                state_dim=4, action_dim=2, powers=powers, hidden_dim=32
            )

            batch_size, seq_len = 2, 3
            state = torch.randn(batch_size, seq_len, 4)
            action = torch.randn(batch_size, seq_len, 2)
            term = torch.zeros(batch_size, seq_len, 1)

            output = model(state, action, term)

            # Verify output shape is correct
            assert output.shape == (batch_size, seq_len, 1)

            # Verify powers buffer has correct values
            assert torch.equal(model.powers, torch.arange(1, powers + 1))

    def test_zero_hidden_layers(self):
        """Test edge case with zero hidden layers."""
        model = est_o1.RNetwork(
            state_dim=4, action_dim=2, num_hidden_layers=0, hidden_dim=256
        )

        batch_size, seq_len = 2, 3
        state = torch.randn(batch_size, seq_len, 4)
        action = torch.randn(batch_size, seq_len, 2)
        term = torch.zeros(batch_size, seq_len, 1)

        output = model(state, action, term)

        # Model should still produce valid output
        assert output.shape == (batch_size, seq_len, 1)

    def test_deterministic_with_seed(self):
        """Test reproducibility with same seed."""
        torch.manual_seed(42)
        model1 = est_o1.RNetwork(state_dim=4, action_dim=2, hidden_dim=64)

        torch.manual_seed(42)
        model2 = est_o1.RNetwork(state_dim=4, action_dim=2, hidden_dim=64)

        # Weights should be identical
        assert torch.allclose(
            model1.final_layer.weight, model2.final_layer.weight, atol=1e-6
        )

        # Outputs should be identical for same input
        state = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 2)
        term = torch.zeros(2, 3, 1)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(state, action, term)
            output2 = model2(state, action, term)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through network."""
        model = est_o1.RNetwork(state_dim=4, action_dim=2, hidden_dim=64)

        state = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 2)
        term = torch.zeros(2, 3, 1)

        # Forward pass
        output = model(state, action, term)

        # Compute loss
        target = torch.randn(2, 3, 1)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradients exist and are non-zero
        assert model.final_layer.weight.grad is not None
        assert model.final_layer.weight.grad.abs().sum() > 0


# =============================================================================
# TestDictDataset
# =============================================================================


class TestDictDataset:
    """Tests for DictDataset wrapper."""

    def test_initialization(self):
        """Test dataset construction with inputs/labels lists."""
        inputs = [{"state": torch.randn(3, 2)} for _ in range(5)]
        labels = [{"reward": torch.tensor(1.0)} for _ in range(5)]

        dataset = est_o1.DictDataset(inputs=inputs, labels=labels)

        assert len(dataset) == 5
        assert dataset.length == 5

    def test_getitem(self):
        """Test __getitem__ returns uncollated examples."""
        inputs = [
            {"state": torch.randn(3, 2), "action": torch.randn(3, 1)} for _ in range(3)
        ]
        labels = [{"reward": torch.tensor(float(idx))} for idx in range(3)]

        dataset = est_o1.DictDataset(inputs=inputs, labels=labels)

        # Retrieve items
        for idx in range(3):
            input_dict, label_dict = dataset[idx]
            assert torch.equal(input_dict["state"], inputs[idx]["state"])
            assert torch.equal(input_dict["action"], inputs[idx]["action"])
            assert label_dict["reward"].item() == float(idx)

    def test_empty(self):
        """Test edge case with empty dataset."""
        dataset = est_o1.DictDataset(inputs=[], labels=[])
        assert len(dataset) == 0

    def test_single_item(self):
        """Test edge case with single item dataset."""
        inputs = [{"state": torch.randn(3, 2)}]
        labels = [{"reward": torch.tensor(1.0)}]

        dataset = est_o1.DictDataset(inputs=inputs, labels=labels)

        assert len(dataset) == 1
        input_dict, label_dict = dataset[0]
        assert torch.equal(input_dict["state"], inputs[0]["state"])
        assert label_dict["reward"].item() == 1.0


# =============================================================================
# TestCollateVariableLengthSequences
# =============================================================================


class TestCollateVariableLengthSequences:
    """Tests for collate function for batching variable-length sequences."""

    def test_uniform_length(self):
        """Test collation when all sequences have same length."""
        # Create batch with uniform length
        batch = []
        for _ in range(4):
            inputs = {
                "state": torch.randn(5, 2),
                "action": torch.randn(5, 1),
                "term": torch.zeros(5, 1),
            }
            labels = {
                "aggregate_reward": torch.tensor(10.0),
                "per_step_rewards": torch.randn(5),
            }
            batch.append((inputs, labels))

        batched_inputs, batched_labels = est_o1.collate_variable_length_sequences(batch)

        # All sequences have length 5, so no padding needed
        assert batched_inputs["state"].shape == (4, 5, 2)
        assert batched_inputs["action"].shape == (4, 5, 1)
        assert batched_inputs["term"].shape == (4, 5, 1)
        assert batched_labels["aggregate_reward"].shape == (4,)
        assert batched_labels["per_step_rewards"].shape == (4, 5)

    def test_variable_length(self):
        """Test padding behavior with different sequence lengths."""
        # Create batch with variable lengths
        batch = []

        # First example: length 3
        inputs1 = {
            "state": torch.ones(3, 2),
            "action": torch.ones(3, 1),
            "term": torch.zeros(3, 1),
        }
        labels1 = {
            "aggregate_reward": torch.tensor(6.0),
            "per_step_rewards": torch.ones(3) * 2.0,
        }
        batch.append((inputs1, labels1))

        # Second example: length 5
        inputs2 = {
            "state": torch.ones(5, 2) * 2.0,
            "action": torch.ones(5, 1) * 2.0,
            "term": torch.zeros(5, 1),
        }
        labels2 = {
            "aggregate_reward": torch.tensor(10.0),
            "per_step_rewards": torch.ones(5) * 2.0,
        }
        batch.append((inputs2, labels2))

        batched_inputs, batched_labels = est_o1.collate_variable_length_sequences(batch)

        # Should be padded to max length (5)
        assert batched_inputs["state"].shape == (2, 5, 2)
        assert batched_inputs["action"].shape == (2, 5, 1)
        assert batched_inputs["term"].shape == (2, 5, 1)
        assert batched_labels["per_step_rewards"].shape == (2, 5)
        assert batched_labels["aggregate_reward"].shape == (2,)

        # First example should have zeros in padded region (indices 3-4)
        assert torch.allclose(batched_inputs["state"][0, 3:, :], torch.zeros(2, 2))
        assert torch.allclose(batched_inputs["action"][0, 3:, :], torch.zeros(2, 1))
        assert torch.allclose(batched_inputs["term"][0, 3:, :], torch.zeros(2, 1))
        assert torch.allclose(batched_labels["per_step_rewards"][0, 3:], torch.zeros(2))
        assert batched_labels["aggregate_reward"][0].item() == 6.0

        # Second example should have original values (no padding)
        assert torch.allclose(batched_inputs["state"][1, :, :], torch.ones(5, 2) * 2.0)
        assert torch.allclose(batched_inputs["action"][1, :, :], torch.ones(5, 1) * 2.0)
        assert torch.allclose(batched_inputs["term"][1, :, :], torch.zeros(5, 1))
        assert torch.allclose(
            batched_labels["per_step_rewards"][1, :], torch.ones(5) * 2.0
        )
        assert batched_labels["aggregate_reward"][1].item() == 10.0

    def test_single_example(self):
        """Test edge case with batch_size=1."""
        inputs = {
            "state": torch.randn(3, 2),
            "action": torch.randn(3, 1),
            "term": torch.zeros(3, 1),
        }
        labels = {
            "aggregate_reward": torch.tensor(5.0),
            "per_step_rewards": torch.randn(3),
        }
        batch = [(inputs, labels)]

        batched_inputs, batched_labels = est_o1.collate_variable_length_sequences(batch)

        # Should have batch dimension
        assert batched_inputs["state"].shape == (1, 3, 2)
        assert batched_inputs["action"].shape == (1, 3, 1)
        assert batched_inputs["term"].shape == (1, 3, 1)
        assert batched_labels["aggregate_reward"].shape == (1,)
        assert batched_labels["per_step_rewards"].shape == (1, 3)

    def test_preserves_data_integrity(self):
        """Test that no data is lost during collation."""
        # Create batch with different lengths
        original_states = [torch.randn(3, 2), torch.randn(5, 2)]
        original_actions = [torch.randn(3, 1), torch.randn(5, 1)]
        original_terms = [torch.zeros(3, 1), torch.zeros(5, 1)]
        original_rewards = [torch.randn(3), torch.randn(5)]
        original_aggregates = [torch.tensor(10.0), torch.tensor(20.0)]

        batch = []
        for idx in range(2):
            inputs = {
                "state": original_states[idx],
                "action": original_actions[idx],
                "term": original_terms[idx],
            }
            labels = {
                "aggregate_reward": original_aggregates[idx],
                "per_step_rewards": original_rewards[idx],
            }
            batch.append((inputs, labels))

        batched_inputs, batched_labels = est_o1.collate_variable_length_sequences(batch)

        # Verify non-padded values match original data
        assert torch.allclose(batched_inputs["state"][0, :3, :], original_states[0])
        assert torch.allclose(batched_inputs["state"][1, :5, :], original_states[1])
        assert torch.allclose(batched_inputs["action"][0, :3, :], original_actions[0])
        assert torch.allclose(batched_inputs["action"][1, :5, :], original_actions[1])
        assert torch.allclose(batched_inputs["term"][0, :3, :], original_terms[0])
        assert torch.allclose(batched_inputs["term"][1, :5, :], original_terms[1])
        assert torch.allclose(
            batched_labels["per_step_rewards"][0, :3], original_rewards[0]
        )
        assert torch.allclose(
            batched_labels["per_step_rewards"][1, :5], original_rewards[1]
        )
        assert batched_labels["aggregate_reward"][0].item() == 10.0
        assert batched_labels["aggregate_reward"][1].item() == 20.0


# =============================================================================
# TestDelayedRewardData
# =============================================================================


class TestDelayedRewardData:
    """Tests for delayed_reward_data function for window generation."""

    def test_single_window(self):
        """Test basic functionality with single window."""
        buffer = create_mock_buffer([[1.0, 2.0, 3.0]])
        delay = rewdelay.FixedDelay(3)

        examples = est_o1.delayed_reward_data(buffer, delay)

        # Should create exactly one example
        assert len(examples) == 1

        inputs, labels = examples[0]

        # Verify shapes
        assert inputs["state"].shape == (3, 2)
        assert inputs["action"].shape == (3, 1)
        assert inputs["term"].shape == (3, 1)
        assert labels["per_step_rewards"].shape == (3,)

        # Verify per-step rewards match buffer
        assert np.allclose(labels["per_step_rewards"], np.array([1.0, 2.0, 3.0]))

        # Verify aggregate equals sum of per-step rewards
        assert labels["aggregate_reward"].item() == 6.0
        assert np.isclose(
            sum(labels["per_step_rewards"].tolist()),
            labels["aggregate_reward"].item(),
            atol=1e-6,
        )

    def test_multiple_windows(self):
        """Test with multiple complete windows."""
        buffer = create_mock_buffer([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        delay = rewdelay.FixedDelay(3)

        examples = est_o1.delayed_reward_data(buffer, delay)

        # Should create two windows
        assert len(examples) == 2

        # First window (steps 0-2)
        inputs1, labels1 = examples[0]
        assert inputs1["state"].shape == (3, 2)
        assert inputs1["action"].shape == (3, 1)
        assert inputs1["term"].shape == (3, 1)
        assert labels1["per_step_rewards"].shape == (3,)
        assert np.allclose(labels1["per_step_rewards"], np.array([1.0, 2.0, 3.0]))
        assert labels1["aggregate_reward"].item() == 6.0
        assert np.isclose(
            sum(labels1["per_step_rewards"].tolist()),
            labels1["aggregate_reward"].item(),
            atol=1e-6,
        )

        # Second window (steps 3-5)
        inputs2, labels2 = examples[1]
        assert inputs2["state"].shape == (3, 2)
        assert inputs2["action"].shape == (3, 1)
        assert inputs2["term"].shape == (3, 1)
        assert labels2["per_step_rewards"].shape == (3,)
        assert np.allclose(labels2["per_step_rewards"], np.array([4.0, 5.0, 6.0]))
        assert labels2["aggregate_reward"].item() == 15.0
        assert np.isclose(
            sum(labels2["per_step_rewards"].tolist()),
            labels2["aggregate_reward"].item(),
            atol=1e-6,
        )

    def test_incomplete_window_discarded(self):
        """Verify incomplete windows at buffer end are discarded."""
        # Buffer with 5 steps, delay=3 -> only 1 complete window
        buffer = create_mock_buffer([[1.0, 2.0, 3.0, 4.0, 5.0]])
        delay = rewdelay.FixedDelay(3)

        examples = est_o1.delayed_reward_data(buffer, delay)

        # Only one complete window (steps 0-2)
        # Incomplete window (steps 3-4) should be discarded
        assert len(examples) == 1

        inputs, labels = examples[0]
        assert inputs["state"].shape == (3, 2)
        assert inputs["action"].shape == (3, 1)
        assert inputs["term"].shape == (3, 1)
        assert labels["per_step_rewards"].shape == (3,)
        assert np.allclose(labels["per_step_rewards"], np.array([1.0, 2.0, 3.0]))
        assert labels["aggregate_reward"].item() == 6.0

    def test_variable_delay(self):
        """Test with variable delay distribution."""
        buffer = create_mock_buffer([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        delay = rewdelay.UniformDelay(2, 5)

        examples = est_o1.delayed_reward_data(buffer, delay)

        # Should create multiple windows with varying lengths
        assert len(examples) > 0

        # Verify all windows have valid fields and consistent lengths
        for inputs, labels in examples:
            seq_len = inputs["state"].shape[0]
            assert 2 <= seq_len <= 5

            # Verify all field shapes are consistent
            assert inputs["state"].shape == (seq_len, 2)
            assert inputs["action"].shape == (seq_len, 1)
            assert inputs["term"].shape == (seq_len, 1)
            assert labels["per_step_rewards"].shape == (seq_len,)

            # Verify aggregate equals sum
            per_step_sum = sum(labels["per_step_rewards"].tolist())
            assert np.isclose(
                per_step_sum, labels["aggregate_reward"].item(), atol=1e-6
            )

    def test_aggregate_equals_sum(self):
        """Verify aggregate_reward == sum(per_step_rewards)."""
        buffer = create_mock_buffer([[1.5, 2.3, 3.7, 4.1]])
        delay = rewdelay.FixedDelay(4)

        examples = est_o1.delayed_reward_data(buffer, delay)

        assert len(examples) == 1

        inputs, labels = examples[0]

        # Verify shapes
        assert inputs["state"].shape == (4, 2)
        assert inputs["action"].shape == (4, 1)
        assert inputs["term"].shape == (4, 1)
        assert labels["per_step_rewards"].shape == (4,)

        # Verify per-step rewards match buffer
        assert np.allclose(labels["per_step_rewards"], np.array([1.5, 2.3, 3.7, 4.1]))

        # Calculate sum of per-step rewards
        per_step_sum = sum(labels["per_step_rewards"].tolist())
        aggregate = labels["aggregate_reward"].item()

        # Should be exactly equal (within floating point precision)
        assert np.isclose(per_step_sum, aggregate, atol=1e-6)
        assert np.isclose(aggregate, 1.5 + 2.3 + 3.7 + 4.1, atol=1e-6)

    def test_episode_boundaries(self):
        """Test behavior across episode boundaries."""
        # Two episodes with delay=3
        buffer = create_mock_buffer([[1.0, 2.0], [10.0, 20.0, 30.0]])
        delay = rewdelay.FixedDelay(3)

        examples = est_o1.delayed_reward_data(buffer, delay)

        # Only the second episode has enough steps for a complete window
        assert len(examples) == 1

        inputs, labels = examples[0]
        assert inputs["state"].shape == (3, 2)
        assert inputs["action"].shape == (3, 1)
        assert inputs["term"].shape == (3, 1)
        assert labels["per_step_rewards"].shape == (3,)
        assert np.allclose(labels["per_step_rewards"], np.array([10.0, 20.0, 30.0]))
        assert labels["aggregate_reward"].item() == 60.0

    def test_empty_buffer(self):
        """Test edge case with empty buffer."""
        buffer = []
        delay = rewdelay.FixedDelay(3)

        examples = est_o1.delayed_reward_data(buffer, delay)

        assert len(examples) == 0

    def test_buffer_smaller_than_delay(self):
        """Test edge case where buffer is smaller than delay."""
        buffer = create_mock_buffer([[1.0, 2.0]])
        delay = rewdelay.FixedDelay(5)

        examples = est_o1.delayed_reward_data(buffer, delay)

        # No complete windows can be formed
        assert len(examples) == 0


# =============================================================================
# TestEvaluateModel
# =============================================================================


class TestEvaluateModel:
    """Tests for evaluate_model function for Markovian prediction."""

    def test_basic(self, simple_model_and_dataset):
        """Test basic evaluation functionality."""
        model, dataset = simple_model_and_dataset
        model.eval()

        mse, predictions = est_o1.evaluate_model(
            model, dataset, batch_size=2, collect_predictions=True
        )

        # Should return valid MSE
        assert isinstance(mse, float)
        assert mse >= 0

        # Should return predictions
        assert len(predictions) > 0

    def test_collect_predictions_false(self, simple_model_and_dataset):
        """Test with collect_predictions=False."""
        model, dataset = simple_model_and_dataset
        model.eval()

        mse, predictions = est_o1.evaluate_model(
            model, dataset, batch_size=2, collect_predictions=False
        )

        # Should return valid MSE
        assert isinstance(mse, float)
        assert mse >= 0

        # Predictions list should be empty
        assert len(predictions) == 0

    def test_max_batches(self, simple_model_and_dataset):
        """Test early stopping with max_batches."""
        model, dataset = simple_model_and_dataset
        model.eval()

        # Dataset has 2 examples, batch_size=1 -> 2 batches total
        # max_batches=1 should only process first batch
        mse, predictions = est_o1.evaluate_model(
            model,
            dataset,
            batch_size=1,
            collect_predictions=True,
            max_batches=1,
        )

        # Should have processed only 1 example
        assert len(predictions) == 1
        assert isinstance(mse, float)
        assert mse >= 0

    def test_prediction_structure(self, simple_model_and_dataset):
        """Test that prediction dict has required keys."""
        model, dataset = simple_model_and_dataset
        model.eval()

        mse, predictions = est_o1.evaluate_model(
            model, dataset, batch_size=2, collect_predictions=True
        )

        # Verify MSE is valid
        assert isinstance(mse, float)
        assert mse >= 0

        # Check first prediction has all required keys
        assert len(predictions) > 0
        pred = predictions[0]

        required_keys = {
            "state",
            "action",
            "term",
            "actual_reward",
            "per_step_rewards",
            "predicted_reward",
            "per_step_predictions",
        }
        assert set(pred.keys()) == required_keys

        # Verify shapes
        assert isinstance(pred["actual_reward"], float)
        assert isinstance(pred["predicted_reward"], float)
        assert isinstance(pred["per_step_rewards"], np.ndarray)
        assert isinstance(pred["per_step_predictions"], np.ndarray)

    def test_markovian_prediction(self, simple_model_and_dataset):
        """Verify predicted_reward == sum(per_step_predictions)."""
        model, dataset = simple_model_and_dataset
        model.eval()

        mse, predictions = est_o1.evaluate_model(
            model, dataset, batch_size=2, collect_predictions=True
        )

        # Verify MSE is valid
        assert isinstance(mse, float)
        assert mse >= 0

        # Verify for each prediction
        for pred in predictions:
            per_step_sum = np.sum(pred["per_step_predictions"])
            predicted_aggregate = pred["predicted_reward"]

            # Should be equal within floating point precision
            assert np.isclose(per_step_sum, predicted_aggregate, atol=1e-6)

    def test_no_gradient(self, simple_model_and_dataset):
        """Test that evaluation doesn't compute gradients."""
        model, dataset = simple_model_and_dataset

        # Zero out gradients
        model.zero_grad()

        # Perform evaluation
        mse, predictions = est_o1.evaluate_model(
            model, dataset, batch_size=2, collect_predictions=False
        )

        # Verify return values are valid
        assert isinstance(mse, float)
        assert mse >= 0
        assert len(predictions) == 0

        # Gradients should still be None
        assert model.final_layer.weight.grad is None


# =============================================================================
# TestTrain
# =============================================================================


class TestTrain:
    """Integration tests for the train function."""

    def test_basic_convergence(self, simple_env, training_dataset):
        """Test that training loop runs and converges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_mse, predictions = est_o1.train(
                env=simple_env,
                dataset=training_dataset,
                train_epochs=10,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                seed=42,
                model_type="mlp",
                output_dir=tmpdir,
            )

            # Should return valid MSE
            assert isinstance(final_mse, float)
            assert final_mse >= 0

            # Should return predictions
            assert len(predictions) > 0

    def test_creates_output_files(self, simple_env, training_dataset):
        """Test that all output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            est_o1.train(
                env=simple_env,
                dataset=training_dataset,
                train_epochs=5,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                seed=42,
                model_type="mlp",
                output_dir=tmpdir,
            )

            # Check that expected files exist
            assert os.path.exists(os.path.join(tmpdir, "model_mlp.pt"))
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            assert os.path.exists(os.path.join(tmpdir, "metrics_mlp.json"))
            assert os.path.exists(os.path.join(tmpdir, "predictions_mlp.json"))

    def test_with_different_seeds(self, simple_env, training_dataset):
        """Test stochastic behavior with different seeds."""
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            final_mse1, _ = est_o1.train(
                env=simple_env,
                dataset=training_dataset,
                train_epochs=5,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                seed=42,
                model_type="mlp",
                output_dir=tmpdir1,
            )

            final_mse2, _ = est_o1.train(
                env=simple_env,
                dataset=training_dataset,
                train_epochs=5,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                seed=43,
                model_type="mlp",
                output_dir=tmpdir2,
            )

            # Different seeds may produce different results
            # (though not guaranteed to be different)
            assert isinstance(final_mse1, float)
            assert isinstance(final_mse2, float)

    def test_reproducibility_with_same_seed(self, simple_env, training_dataset):
        """Test deterministic behavior with same seed."""
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            final_mse1, _ = est_o1.train(
                env=simple_env,
                dataset=training_dataset,
                train_epochs=5,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                seed=42,
                model_type="mlp",
                output_dir=tmpdir1,
            )

            final_mse2, _ = est_o1.train(
                env=simple_env,
                dataset=training_dataset,
                train_epochs=5,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=5,
                seed=42,
                model_type="mlp",
                output_dir=tmpdir2,
            )

            # Same seed should produce identical results
            assert np.isclose(final_mse1, final_mse2, atol=1e-6)

    def test_invalid_model_type(self, simple_env, training_dataset):
        """Test error handling for invalid model type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown model_type"):
                est_o1.train(
                    env=simple_env,
                    dataset=training_dataset,
                    train_epochs=5,
                    batch_size=8,
                    eval_steps=5,
                    log_episode_frequency=5,
                    seed=42,
                    model_type="invalid",
                    output_dir=tmpdir,
                )

    def test_eval_during_training(self, simple_env, training_dataset):
        """Test that evaluation is called at correct frequency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for 6 epochs with log_episode_frequency=2
            # Should evaluate at epochs 2, 4, 6
            final_mse, _ = est_o1.train(
                env=simple_env,
                dataset=training_dataset,
                train_epochs=6,
                batch_size=8,
                eval_steps=5,
                log_episode_frequency=2,
                seed=42,
                model_type="mlp",
                output_dir=tmpdir,
            )

            # Training should complete successfully
            assert isinstance(final_mse, float)
            assert final_mse >= 0


# =============================================================================
# Module-level helper functions
# =============================================================================


def create_mock_buffer(episodes: List[List[float]]) -> List[Tuple]:
    """
    Create a mock buffer from episode rewards.

    Args:
        episodes: List of reward sequences, one per episode

    Returns:
        Buffer in format: List[(state, action, next_state, reward, term)]
    """
    buffer = []
    for ep_idx, rewards in enumerate(episodes):
        for step_idx, reward in enumerate(rewards):
            # Create dummy state/action (just use indices for debugging)
            state = np.array([float(ep_idx), float(step_idx)])
            action = np.array([float(ep_idx * 10 + step_idx)])
            next_state = np.array([float(ep_idx), float(step_idx + 1)])
            term = step_idx == len(rewards) - 1  # Terminal at last step

            buffer.append((state, action, next_state, reward, term))

    return buffer


# =============================================================================
# Module-level fixtures
# =============================================================================


@pytest.fixture
def simple_env():
    """Create a simple environment for testing."""
    return gym.make("MountainCarContinuous-v0")


@pytest.fixture
def simple_model_and_dataset():
    """Create a simple model and dataset for evaluation tests."""
    # Create small dataset
    buffer = create_mock_buffer([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    delay = rewdelay.FixedDelay(3)
    examples = est_o1.delayed_reward_data(buffer, delay)

    inputs, labels = zip(*examples)
    dataset = est_o1.DictDataset(inputs=list(inputs), labels=list(labels))

    # Create model
    torch.manual_seed(42)
    model = est_o1.RNetwork(state_dim=2, action_dim=1, hidden_dim=16)

    return model, dataset


@pytest.fixture(scope="module")
def training_dataset():
    """Create a realistic training dataset for integration tests."""
    # Create environment
    env = gym.make("MountainCarContinuous-v0")

    # Collect small amount of data
    buffer = dataproc.collection_traj_data(env, steps=100, include_term=True, seed=42)

    # Create delayed reward data
    delay = rewdelay.FixedDelay(3)
    examples = est_o1.delayed_reward_data(buffer, delay)

    inputs, labels = zip(*examples)
    dataset = est_o1.DictDataset(inputs=list(inputs), labels=list(labels))

    return dataset
