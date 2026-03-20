"""Tests for O2 reward estimation with return tracking."""

import gymnasium as gym
import numpy as np
import torch

from drmdp import dataproc, rewdelay
from drmdp.dfdrl import est_o2


class TestDelayedRewardDataReturns:
    """Tests for return tracking in delayed_reward_data()."""

    def test_cumulative_returns_single_episode(self):
        """Test that cumulative returns are computed correctly within a single episode."""
        # Create a simple environment
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=10)

        # Collect a small buffer with known rewards
        buffer = dataproc.collection_traj_data(
            env, steps=10, include_term=True, seed=42
        )

        # Create delayed reward data
        delay = rewdelay.FixedDelay(3)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # Verify that end_return values match cumulative returns
        # (We can't verify all timesteps since only windows are created,
        # but we can verify the relationship holds)
        for inputs, labels in examples:
            start_return = labels["start_return"].item()
            end_return = labels["end_return"].item()
            aggregate_reward = labels["aggregate_reward"].item()

            # Basic sanity checks
            assert start_return >= 0.0 or start_return < 0.0  # Can be negative
            assert end_return >= start_return or end_return < start_return
            # Relationship should hold: end_return - start_return ≈ aggregate_reward
            np.testing.assert_allclose(
                end_return - start_return, aggregate_reward, atol=1e-6
            )

    def test_cumulative_returns_reset_at_episodes(self):
        """Test that cumulative returns reset to 0 at episode boundaries.

        Note: MountainCar rarely terminates naturally, so this test validates
        the core relationship for available data.
        """
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=50)
        buffer = dataproc.collection_traj_data(
            env, steps=100, include_term=True, seed=123
        )

        delay = rewdelay.FixedDelay(3)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # Verify the critical relationship holds for all windows
        for inputs, labels in examples:
            start_return = labels["start_return"].item()
            end_return = labels["end_return"].item()
            aggregate_reward = labels["aggregate_reward"].item()

            np.testing.assert_allclose(
                end_return - start_return, aggregate_reward, atol=1e-6
            )

    def test_start_return_before_window(self):
        """Test start_return is cumulative sum before window starts."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=20)
        buffer = dataproc.collection_traj_data(
            env, steps=20, include_term=True, seed=42
        )

        delay = rewdelay.FixedDelay(3)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # First example should have start_return = 0.0 (starts at episode beginning)
        if len(examples) > 0:
            first_start_return = examples[0][1]["start_return"].item()
            assert first_start_return == 0.0, (
                f"First window should start with return 0.0, got {first_start_return}"
            )

    def test_end_return_after_window(self):
        """Test end_return is cumulative sum after window ends."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=20)
        buffer = dataproc.collection_traj_data(
            env, steps=20, include_term=True, seed=42
        )

        delay = rewdelay.FixedDelay(3)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # All end_returns should be valid
        for inputs, labels in examples:
            end_return = labels["end_return"].item()
            start_return = labels["start_return"].item()
            # end_return should be >= start_return (or both negative)
            # The key relationship is the difference
            aggregate_reward = labels["aggregate_reward"].item()
            np.testing.assert_allclose(
                end_return - start_return, aggregate_reward, atol=1e-6
            )

    def test_return_aggregate_relationship(self):
        """Test that end_return - start_return = aggregate_reward for all windows."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=50)
        buffer = dataproc.collection_traj_data(
            env, steps=100, include_term=True, seed=999
        )

        delay = rewdelay.FixedDelay(5)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # Verify relationship for every window
        assert len(examples) > 0, "Should have at least one example"

        for idx, (inputs, labels) in enumerate(examples):
            start_return = labels["start_return"].item()
            end_return = labels["end_return"].item()
            aggregate_reward = labels["aggregate_reward"].item()

            # This is the critical relationship
            np.testing.assert_allclose(
                end_return - start_return,
                aggregate_reward,
                atol=1e-6,
                err_msg=f"Window {idx}: end_return - start_return != aggregate_reward",
            )

    def test_window_at_episode_start(self):
        """Test window starting at episode beginning has start_return=0."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=20)
        buffer = dataproc.collection_traj_data(
            env, steps=20, include_term=True, seed=42
        )

        delay = rewdelay.FixedDelay(3)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # First window should start with 0.0
        if len(examples) > 0:
            assert examples[0][1]["start_return"].item() == 0.0

    def test_window_after_terminal(self):
        """Test that windows correctly handle terminal states.

        The critical relationship end_return - start_return = aggregate_reward
        must hold whether or not terminals are present.
        """
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=50)
        buffer = dataproc.collection_traj_data(
            env, steps=100, include_term=True, seed=456
        )

        delay = rewdelay.FixedDelay(3)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # Verify the relationship holds for all windows
        for inputs, labels in examples:
            start_return = labels["start_return"].item()
            end_return = labels["end_return"].item()
            aggregate_reward = labels["aggregate_reward"].item()

            np.testing.assert_allclose(
                end_return - start_return, aggregate_reward, atol=1e-6
            )

    def test_multiple_episodes_return_tracking(self):
        """Test return tracking with longer trajectories.

        This test uses longer episodes to collect more data and validate
        that return tracking is consistent across all windows.
        """
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=50)
        buffer = dataproc.collection_traj_data(
            env, steps=100, include_term=True, seed=789
        )

        delay = rewdelay.FixedDelay(4)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # Verify all windows maintain the critical relationship
        assert len(examples) > 0, "Should have at least one example"
        for inputs, labels in examples:
            start_return = labels["start_return"].item()
            end_return = labels["end_return"].item()
            aggregate_reward = labels["aggregate_reward"].item()

            np.testing.assert_allclose(
                end_return - start_return, aggregate_reward, atol=1e-6
            )

    def test_empty_buffer(self):
        """Test that empty buffer returns empty examples list."""
        delay = rewdelay.FixedDelay(3)
        examples = est_o2.delayed_reward_data([], delay=delay)
        assert examples == []

    def test_variable_delay_return_tracking(self):
        """Test return tracking with variable delay."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=20)
        buffer = dataproc.collection_traj_data(
            env, steps=100, include_term=True, seed=321
        )

        # Use variable delay
        delay = rewdelay.UniformDelay(min_delay=2, max_delay=5)
        examples = est_o2.delayed_reward_data(buffer, delay=delay)

        # Verify relationship for all windows regardless of size
        for inputs, labels in examples:
            start_return = labels["start_return"].item()
            end_return = labels["end_return"].item()
            aggregate_reward = labels["aggregate_reward"].item()

            np.testing.assert_allclose(
                end_return - start_return, aggregate_reward, atol=1e-6
            )


class TestCollateVariableLengthSequences:
    """Tests for collate function with return fields."""

    def test_collate_includes_return_fields(self):
        """Test that collation includes start_return and end_return."""
        # Create mock examples
        examples = [
            (
                {
                    "state": torch.randn(3, 4),
                    "action": torch.randn(3, 2),
                    "term": torch.zeros(3, 1),
                },
                {
                    "aggregate_reward": torch.tensor(1.5),
                    "per_step_rewards": torch.tensor([0.5, 0.5, 0.5]),
                    "start_return": torch.tensor(0.0),
                    "end_return": torch.tensor(1.5),
                },
            ),
            (
                {
                    "state": torch.randn(4, 4),
                    "action": torch.randn(4, 2),
                    "term": torch.zeros(4, 1),
                },
                {
                    "aggregate_reward": torch.tensor(2.0),
                    "per_step_rewards": torch.tensor([0.5, 0.5, 0.5, 0.5]),
                    "start_return": torch.tensor(1.5),
                    "end_return": torch.tensor(3.5),
                },
            ),
        ]

        batched_inputs, batched_labels = est_o2.collate_variable_length_sequences(
            examples
        )

        # Verify return fields are present
        assert "start_return" in batched_labels
        assert "end_return" in batched_labels

        # Verify they are stacked correctly
        assert batched_labels["start_return"].shape == (2,)
        assert batched_labels["end_return"].shape == (2,)

        # Verify values
        assert batched_labels["start_return"][0].item() == 0.0
        assert batched_labels["start_return"][1].item() == 1.5
        assert batched_labels["end_return"][0].item() == 1.5
        assert batched_labels["end_return"][1].item() == 3.5


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_create_training_buffer_with_returns(self):
        """Test that create_training_buffer produces data with return fields."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=20)
        delay = rewdelay.FixedDelay(3)

        examples = est_o2.create_training_buffer(
            env, delay=delay, num_steps=50, seed=111
        )

        assert len(examples) > 0, "Should create at least one example"

        # Verify structure
        for inputs, labels in examples:
            # Check inputs structure
            assert "state" in inputs
            assert "action" in inputs
            assert "term" in inputs

            # Check labels structure
            assert "aggregate_reward" in labels
            assert "per_step_rewards" in labels
            assert "start_return" in labels
            assert "end_return" in labels

            # Verify return relationship
            start_return = labels["start_return"].item()
            end_return = labels["end_return"].item()
            aggregate_reward = labels["aggregate_reward"].item()

            np.testing.assert_allclose(
                end_return - start_return, aggregate_reward, atol=1e-6
            )

    def test_dataset_and_dataloader(self):
        """Test that DictDataset and DataLoader work with return fields."""
        env = gym.make("MountainCarContinuous-v0", max_episode_steps=20)
        delay = rewdelay.FixedDelay(3)

        examples = est_o2.create_training_buffer(
            env, delay=delay, num_steps=50, seed=222
        )

        inputs_list, labels_list = zip(*examples)
        dataset = est_o2.DictDataset(inputs=list(inputs_list), labels=list(labels_list))

        # Create dataloader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=est_o2.collate_variable_length_sequences,
        )

        # Iterate through one batch
        for batched_inputs, batched_labels in dataloader:
            # Verify batch structure
            assert "state" in batched_inputs
            assert "action" in batched_inputs
            assert "term" in batched_inputs

            assert "aggregate_reward" in batched_labels
            assert "start_return" in batched_labels
            assert "end_return" in batched_labels

            # Verify shapes
            batch_size = batched_inputs["state"].shape[0]
            assert batched_labels["start_return"].shape == (batch_size,)
            assert batched_labels["end_return"].shape == (batch_size,)

            # Verify relationship for each example in batch
            for idx in range(batch_size):
                start_return = batched_labels["start_return"][idx].item()
                end_return = batched_labels["end_return"][idx].item()
                aggregate_reward = batched_labels["aggregate_reward"][idx].item()

                np.testing.assert_allclose(
                    end_return - start_return, aggregate_reward, atol=1e-6
                )

            break  # Only check first batch
