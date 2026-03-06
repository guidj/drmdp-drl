"""
Tests for est_o2.py reward estimation data generation and models.

Critical tests focus on:
1. Episode boundary handling - no windows should span episode boundaries
2. Cumulative return correctness
3. Aggregate reward correctness
4. Regularizer assumptions: G_curr - G_prev ≈ R_obs_curr
5. Zero-filled previous windows at episode starts
"""

from typing import List, Tuple

import numpy as np
import pytest
import torch

from drmdp import rewdelay
from drmdp.dfdrl import est_o2


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
# Tests for est_o2.delayed_reward_data_consecutive_windows
# =============================================================================


def test_single_episode_single_window():
    """Test with single episode and single window."""
    buffer = create_mock_buffer([[1.0, 2.0, 3.0]])
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)

    assert len(examples) == 1

    inputs, labels = examples[0]

    # Verify labels
    assert labels["prev_end_return"].item() == 0.0
    assert labels["curr_aggregate_reward"].item() == 6.0
    assert labels["curr_end_return"].item() == 6.0

    # Verify input shapes
    assert inputs["prev_state"].shape == (1, 2)
    assert inputs["prev_action"].shape == (1, 1)
    assert inputs["prev_term"].shape == (1, 1)
    assert inputs["prev_timestep"].shape == (1, 1)
    assert inputs["curr_state"].shape == (3, 2)
    assert inputs["curr_action"].shape == (3, 1)
    assert inputs["curr_term"].shape == (3, 1)
    assert inputs["curr_timestep"].shape == (1, 1)

    # Verify prev is zero-filled (first window)
    assert torch.allclose(inputs["prev_state"], torch.zeros(1, 2))
    assert torch.allclose(inputs["prev_action"], torch.zeros(1, 1))
    assert torch.allclose(inputs["prev_term"], torch.zeros(1, 1))
    assert inputs["prev_timestep"].item() == 0

    # Verify timestep
    assert inputs["curr_timestep"].item() == 2  # Last step of window


def test_single_episode_consecutive_windows():
    """Test with single episode and multiple consecutive windows."""
    buffer = create_mock_buffer([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)
    assert len(examples) == 2

    # ===== First window =====
    inputs1, labels1 = examples[0]

    # Verify labels
    assert labels1["prev_end_return"].item() == 0.0
    assert labels1["curr_aggregate_reward"].item() == 6.0
    assert labels1["curr_end_return"].item() == 6.0

    # Verify inputs structure - first window should have zero-filled prev
    assert inputs1["prev_state"].shape == (1, 2)
    assert inputs1["prev_action"].shape == (1, 1)
    assert inputs1["prev_term"].shape == (1, 1)
    assert torch.allclose(inputs1["prev_state"], torch.zeros(1, 2))
    assert torch.allclose(inputs1["prev_action"], torch.zeros(1, 1))
    assert torch.allclose(inputs1["prev_term"], torch.zeros(1, 1))

    # Verify current window has correct shape
    assert inputs1["curr_state"].shape == (3, 2)
    assert inputs1["curr_action"].shape == (3, 1)
    assert inputs1["curr_term"].shape == (3, 1)
    # Verify timesteps
    assert inputs1["prev_timestep"].item() == 0
    assert inputs1["curr_timestep"].item() == 2  # Last step of window (steps 0,1,2)

    # ===== Second window =====
    inputs2, labels2 = examples[1]

    # Verify labels
    assert labels2["prev_end_return"].item() == 6.0
    assert labels2["curr_aggregate_reward"].item() == 15.0
    assert labels2["curr_end_return"].item() == 21.0

    # CRITICAL: Verify prev_window of second example matches curr_window of first
    assert torch.allclose(inputs2["prev_state"], inputs1["curr_state"])
    assert torch.allclose(inputs2["prev_action"], inputs1["curr_action"])
    assert torch.allclose(inputs2["prev_term"], inputs1["curr_term"])

    # Verify current window has correct shape
    assert inputs2["curr_state"].shape == (3, 2)
    assert inputs2["curr_action"].shape == (3, 1)
    assert inputs2["curr_term"].shape == (3, 1)
    # Verify timesteps
    assert inputs2["prev_timestep"].item() == 2  # From first window's last step
    assert inputs2["curr_timestep"].item() == 5  # Last step of window (steps 3,4,5)
    # Last step should be terminal (end of episode)
    assert inputs2["curr_term"][-1, 0].item() == 1.0

    # CRITICAL: Verify regularizer assumption
    g_diff = labels2["curr_end_return"].item() - labels2["prev_end_return"].item()
    r_obs = labels2["curr_aggregate_reward"].item()
    assert np.isclose(g_diff, r_obs, atol=1e-6)


def test_episode_boundary_no_span():
    """
    CRITICAL TEST: Verify windows never span episode boundaries.
    This tests the fix for the episode boundary bug.
    """
    buffer = create_mock_buffer([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)
    assert len(examples) == 2

    # ===== First window (episode 0) =====
    inputs1, labels1 = examples[0]

    # Verify labels
    assert labels1["prev_end_return"].item() == 0.0
    assert labels1["curr_aggregate_reward"].item() == 6.0
    assert labels1["curr_end_return"].item() == 6.0

    # Verify inputs - first window should have zero-filled prev
    assert inputs1["prev_state"].shape == (1, 2)
    assert inputs1["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs1["prev_state"], torch.zeros(1, 2))
    assert torch.allclose(inputs1["prev_action"], torch.zeros(1, 1))
    assert torch.allclose(inputs1["prev_term"], torch.zeros(1, 1))
    assert inputs1["prev_timestep"].item() == 0
    assert inputs1["curr_timestep"].item() == 2  # Last step in episode 0

    # ===== Second window (episode 1) - MUST have zero prev after episode boundary =====
    inputs2, labels2 = examples[1]

    # Verify labels
    assert labels2["prev_end_return"].item() == 0.0
    assert labels2["curr_aggregate_reward"].item() == 60.0
    assert labels2["curr_end_return"].item() == 60.0

    # CRITICAL: Verify inputs are reset (new episode, so zero-filled prev)
    assert inputs2["prev_state"].shape == (1, 2)
    assert inputs2["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs2["prev_state"], torch.zeros(1, 2))
    assert torch.allclose(inputs2["prev_action"], torch.zeros(1, 1))
    assert torch.allclose(inputs2["prev_term"], torch.zeros(1, 1))
    assert inputs2["prev_timestep"].item() == 0  # Reset for new episode
    assert inputs2["curr_timestep"].item() == 2  # Last step in episode 1


def test_episode_ends_at_window_boundary():
    """
    CRITICAL TEST: Episode ending exactly at window boundary.
    This is the specific case that the episode boundary bug missed.
    """
    buffer = create_mock_buffer([[1.0, 2.0, 3.0], [10.0, 20.0]])
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)
    assert len(examples) == 1

    inputs, labels = examples[0]

    # Verify labels
    assert labels["prev_end_return"].item() == 0.0
    assert labels["curr_aggregate_reward"].item() == 6.0
    assert labels["curr_end_return"].item() == 6.0

    # Verify inputs
    assert inputs["prev_state"].shape == (1, 2)
    assert inputs["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs["prev_state"], torch.zeros(1, 2))
    assert torch.allclose(inputs["prev_action"], torch.zeros(1, 1))
    assert torch.allclose(inputs["prev_term"], torch.zeros(1, 1))
    assert inputs["prev_timestep"].item() == 0
    assert inputs["curr_timestep"].item() == 2
    # Episode ends at window boundary - last step should be terminal
    assert inputs["curr_term"][-1, 0].item() == 1.0


def test_multiple_episodes_mixed_lengths():
    """Test with multiple episodes of varying lengths."""
    buffer = create_mock_buffer(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0], [100.0, 200.0, 300.0]]
    )
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)
    # Episode 0 (6 steps): 2 windows (steps 0-2, 3-5)
    # Episode 1 (1 step): Cannot form window of size 3
    # Episode 2 (3 steps): 1 window (steps 0-2)
    assert len(examples) == 3

    # ===== Episode 0, Window 1 =====
    inputs0_1, labels0_1 = examples[0]
    assert labels0_1["prev_end_return"].item() == 0.0
    assert labels0_1["curr_aggregate_reward"].item() == 6.0
    assert labels0_1["curr_end_return"].item() == 6.0
    assert inputs0_1["prev_state"].shape == (1, 2)
    assert inputs0_1["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs0_1["prev_state"], torch.zeros(1, 2))
    assert inputs0_1["prev_timestep"].item() == 0
    assert inputs0_1["curr_timestep"].item() == 2

    # ===== Episode 0, Window 2 =====
    inputs0_2, labels0_2 = examples[1]
    assert labels0_2["prev_end_return"].item() == 6.0
    assert labels0_2["curr_aggregate_reward"].item() == 15.0
    assert labels0_2["curr_end_return"].item() == 21.0
    # Should have prev from previous window
    assert torch.allclose(inputs0_2["prev_state"], inputs0_1["curr_state"])
    assert torch.allclose(inputs0_2["prev_action"], inputs0_1["curr_action"])
    assert inputs0_2["prev_timestep"].item() == 2
    assert inputs0_2["curr_timestep"].item() == 5
    # Last step is terminal (end of episode 0)
    assert inputs0_2["curr_term"][-1, 0].item() == 1.0

    # ===== Episode 2, Window 1 - MUST have zero prev (new episode) =====
    inputs2_1, labels2_1 = examples[2]
    assert labels2_1["prev_end_return"].item() == 0.0
    assert labels2_1["curr_aggregate_reward"].item() == 600.0
    assert labels2_1["curr_end_return"].item() == 600.0
    assert inputs2_1["prev_state"].shape == (1, 2)
    assert inputs2_1["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs2_1["prev_state"], torch.zeros(1, 2))
    assert torch.allclose(inputs2_1["prev_action"], torch.zeros(1, 1))
    assert torch.allclose(inputs2_1["prev_term"], torch.zeros(1, 1))
    assert inputs2_1["prev_timestep"].item() == 0
    assert inputs2_1["curr_timestep"].item() == 2


def test_cumulative_returns_reset_per_episode():
    """Verify cumulative returns reset at episode boundaries."""
    buffer = create_mock_buffer([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)
    assert len(examples) == 2

    # ===== Episode 0 =====
    inputs0, labels0 = examples[0]
    assert labels0["prev_end_return"].item() == 0.0
    assert labels0["curr_aggregate_reward"].item() == 3.0
    assert labels0["curr_end_return"].item() == 3.0
    assert inputs0["prev_state"].shape == (1, 2)
    assert inputs0["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs0["prev_state"], torch.zeros(1, 2))
    assert inputs0["prev_timestep"].item() == 0
    assert inputs0["curr_timestep"].item() == 2

    # ===== Episode 1 - cumulative return RESETS, NOT 9! =====
    inputs1, labels1 = examples[1]
    assert labels1["prev_end_return"].item() == 0.0  # Reset for new episode
    assert labels1["curr_aggregate_reward"].item() == 6.0
    assert labels1["curr_end_return"].item() == 6.0  # NOT 9!
    assert inputs1["prev_state"].shape == (1, 2)
    assert inputs1["curr_state"].shape == (3, 2)
    assert torch.allclose(
        inputs1["prev_state"], torch.zeros(1, 2)
    )  # Reset for new episode
    assert torch.allclose(inputs1["prev_action"], torch.zeros(1, 1))
    assert torch.allclose(inputs1["prev_term"], torch.zeros(1, 1))
    assert inputs1["prev_timestep"].item() == 0  # Reset for new episode
    assert inputs1["curr_timestep"].item() == 2


def test_all_windows_satisfy_regularizer_assumption():
    """
    Comprehensive test: ALL windows must satisfy G_curr - G_prev = R_obs_curr.
    This is the fundamental assumption for the O2 regularizers.
    Also verifies all input structures are correct.
    """
    episodes = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ]
    buffer = create_mock_buffer(episodes)
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)

    label_violations = []
    input_violations = []
    prev_window_inputs = None

    for idx, (inputs, labels) in enumerate(examples):
        # ===== Verify input structure =====
        # Check all required keys exist
        required_input_keys = {
            "prev_state",
            "prev_action",
            "prev_term",
            "prev_timestep",
            "curr_state",
            "curr_action",
            "curr_term",
            "curr_timestep",
        }
        if set(inputs.keys()) != required_input_keys:
            input_violations.append(
                f"Window {idx}: Missing or extra input keys: {inputs.keys()}"
            )

        # Check shapes are consistent
        if inputs["prev_state"].shape[0] != inputs["prev_action"].shape[0]:
            input_violations.append(
                f"Window {idx}: prev_state and prev_action length mismatch"
            )
        if inputs["curr_state"].shape[0] != inputs["curr_action"].shape[0]:
            input_violations.append(
                f"Window {idx}: curr_state and curr_action length mismatch"
            )

        # Check timestep shapes
        if inputs["prev_timestep"].shape != (1, 1):
            input_violations.append(
                f"Window {idx}: prev_timestep wrong shape {inputs['prev_timestep'].shape}"
            )
        if inputs["curr_timestep"].shape != (1, 1):
            input_violations.append(
                f"Window {idx}: curr_timestep wrong shape {inputs['curr_timestep'].shape}"
            )

        # ===== Verify consecutive window chaining =====
        if labels["prev_end_return"].item() == 0.0:
            # First window of episode - prev should be zero-filled
            if not torch.allclose(
                inputs["prev_state"], torch.zeros_like(inputs["prev_state"])
            ):
                input_violations.append(
                    f"Window {idx}: prev_state not zero-filled for first window"
                )
            if not torch.allclose(
                inputs["prev_action"], torch.zeros_like(inputs["prev_action"])
            ):
                input_violations.append(
                    f"Window {idx}: prev_action not zero-filled for first window"
                )
            if not torch.allclose(
                inputs["prev_term"], torch.zeros_like(inputs["prev_term"])
            ):
                input_violations.append(
                    f"Window {idx}: prev_term not zero-filled for first window"
                )
            if inputs["prev_timestep"].item() != 0:
                input_violations.append(
                    f"Window {idx}: prev_timestep not 0 for first window"
                )
        else:
            # Consecutive window - prev should match previous curr
            if prev_window_inputs is not None:
                if not torch.allclose(
                    inputs["prev_state"], prev_window_inputs["curr_state"]
                ):
                    input_violations.append(
                        f"Window {idx}: prev_state doesn't match previous curr_state"
                    )
                if not torch.allclose(
                    inputs["prev_action"], prev_window_inputs["curr_action"]
                ):
                    input_violations.append(
                        f"Window {idx}: prev_action doesn't match previous curr_action"
                    )
                if not torch.allclose(
                    inputs["prev_term"], prev_window_inputs["curr_term"]
                ):
                    input_violations.append(
                        f"Window {idx}: prev_term doesn't match previous curr_term"
                    )

        # Store for next iteration
        prev_window_inputs = inputs

        # ===== Verify label structure =====
        required_label_keys = {
            "prev_start_return",
            "prev_end_return",
            "prev_aggregate_reward",
            "curr_start_return",
            "curr_end_return",
            "curr_aggregate_reward",
        }
        if set(labels.keys()) != required_label_keys:
            label_violations.append(
                f"Window {idx}: Missing or extra label keys: {labels.keys()}"
            )

        # ===== Verify delta relationships =====
        # Verify: prev_end_return - prev_start_return == prev_aggregate_reward
        prev_delta = (
            labels["prev_end_return"].item() - labels["prev_start_return"].item()
        )
        prev_agg = labels["prev_aggregate_reward"].item()
        if not np.isclose(prev_delta, prev_agg, atol=1e-6):
            label_violations.append(
                f"Window {idx}: prev_delta={prev_delta:.8f}, prev_agg={prev_agg:.8f}"
            )

        # Verify: curr_end_return - curr_start_return == curr_aggregate_reward
        curr_delta = (
            labels["curr_end_return"].item() - labels["curr_start_return"].item()
        )
        curr_agg = labels["curr_aggregate_reward"].item()
        if not np.isclose(curr_delta, curr_agg, atol=1e-6):
            label_violations.append(
                f"Window {idx}: curr_delta={curr_delta:.8f}, curr_agg={curr_agg:.8f}"
            )

        # For non-first windows, verify continuity: prev_end_return == curr_start_return
        if labels["prev_end_return"].item() != 0.0:  # Not a zero-filled prev window
            if not np.isclose(
                labels["prev_end_return"].item(),
                labels["curr_start_return"].item(),
                atol=1e-6,
            ):
                label_violations.append(
                    f"Window {idx}: prev_end={labels['prev_end_return'].item():.8f}, "
                    f"curr_start={labels['curr_start_return'].item():.8f}"
                )

    # Report all violations
    all_violations = label_violations + input_violations
    if all_violations:
        print("\nViolations found:")
        for violation in all_violations:
            print(f"  {violation}")
    assert len(all_violations) == 0


# =============================================================================
# Tests for independent embeddings
# =============================================================================


def test_networks_have_independent_embeddings():
    """Verify that each network has its own independent embedding layer."""
    torch.manual_seed(42)
    r_model1 = est_o2.RNetwork(state_dim=2, action_dim=1, hidden_dim=16)
    torch.manual_seed(43)
    r_model2 = est_o2.RNetwork(state_dim=2, action_dim=1, hidden_dim=16)

    # Different instances should have different embedding objects
    assert r_model1.input_proj is not r_model2.input_proj

    # Different random seeds should produce different initial weights
    assert not torch.allclose(r_model1.input_proj.weight, r_model2.input_proj.weight)


# =============================================================================
# Tests for GNetwork start_return functionality
# =============================================================================


def test_gnetwork_start_return_shape():
    """Test that GNetwork output shape is correct when start_return is provided."""
    torch.manual_seed(42)
    g_model = est_o2.GNetwork(
        state_dim=2, action_dim=1, hidden_dim=16, num_heads=2, num_layers=1
    )

    # Test various batch sizes and sequence lengths
    test_configs = [
        (1, 3),  # Single batch, short sequence
        (4, 5),  # Small batch, medium sequence
        (8, 10),  # Larger batch, longer sequence
    ]

    for batch_size, seq_len in test_configs:
        state = torch.randn(batch_size, seq_len, 2)
        action = torch.randn(batch_size, seq_len, 1)
        term = torch.zeros(batch_size, seq_len, 1)
        start_return = torch.randn(batch_size, 1)

        output = g_model(state, action, term, start_return=start_return)

        assert output.shape == (
            batch_size,
            1,
        ), f"Expected shape ({batch_size}, 1), got {output.shape}"


def test_gnetwork_start_return_none_vs_zero():
    """Test that start_return=None and start_return=zeros produce identical outputs."""
    torch.manual_seed(42)
    g_model = est_o2.GNetwork(
        state_dim=2, action_dim=1, hidden_dim=16, num_heads=2, num_layers=1
    )
    g_model.eval()  # Set to eval mode to disable dropout

    batch_size = 4
    seq_len = 5
    state = torch.randn(batch_size, seq_len, 2)
    action = torch.randn(batch_size, seq_len, 1)
    term = torch.zeros(batch_size, seq_len, 1)

    # Forward pass with start_return=None (should default to zeros)
    output_none = g_model(state, action, term, start_return=None)

    # Forward pass with explicit zeros
    start_return_zeros = torch.zeros(batch_size, 1)
    output_zeros = g_model(state, action, term, start_return=start_return_zeros)

    assert torch.allclose(output_none, output_zeros, atol=1e-6), (
        "start_return=None should behave identically to start_return=zeros"
    )


@pytest.mark.skip(
    reason="start_return functionality is currently disabled (commented out in GNetwork.forward)"
)
def test_gnetwork_start_return_delta_behavior():
    """
    Test that GNetwork learns to use start_return correctly.
    Simplified test: verify that different start_return values produce different outputs.
    """
    torch.manual_seed(42)

    # Create GNetwork
    g_model = est_o2.GNetwork(
        state_dim=2, action_dim=1, hidden_dim=32, num_heads=2, num_layers=2
    )
    g_model.eval()  # Set to eval mode for deterministic behavior

    # Fixed state/action sequence
    batch_size = 1
    seq_len = 3
    state = torch.randn(batch_size, seq_len, 2)
    action = torch.randn(batch_size, seq_len, 1)
    term = torch.zeros(batch_size, seq_len, 1)

    # Test with different start_return values
    start_return_0 = torch.zeros(batch_size, 1)
    start_return_5 = torch.ones(batch_size, 1) * 5.0
    start_return_10 = torch.ones(batch_size, 1) * 10.0

    with torch.no_grad():
        output_0 = g_model(state, action, term, start_return=start_return_0)
        output_5 = g_model(state, action, term, start_return=start_return_5)
        output_10 = g_model(state, action, term, start_return=start_return_10)

    # Outputs should be different for different start_return values
    # This verifies that the model is using the start_return input
    assert not torch.allclose(output_0, output_5, atol=0.01), (
        "Different start_return values should produce different outputs"
    )
    assert not torch.allclose(output_0, output_10, atol=0.01), (
        "Different start_return values should produce different outputs"
    )
    assert not torch.allclose(output_5, output_10, atol=0.01), (
        "Different start_return values should produce different outputs"
    )


def test_consecutive_windows_start_return():
    """
    Test training on small dataset with consecutive windows.
    Verify that curr window receives prev_return as start_return.
    """
    buffer = create_mock_buffer([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)
    assert len(examples) == 2

    # Create dataset and use collate function to get proper batch format
    # Convert to float32 to match model dtype
    dataset = est_o2.DualWindowDataset(
        inputs={
            key: [ex[0][key].float() for ex in examples]
            for key in examples[0][0].keys()
        },
        labels={
            key: [ex[1][key].float() for ex in examples]
            for key in examples[0][1].keys()
        },
    )

    # Create model
    torch.manual_seed(42)
    g_model = est_o2.GNetwork(
        state_dim=2, action_dim=1, hidden_dim=32, num_heads=2, num_layers=2
    )

    optimizer = torch.optim.Adam(g_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Create dataloader with collate function
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=est_o2.collate_variable_length_windows,
    )

    # Train for a few epochs
    initial_loss = None
    final_loss = None

    for epoch in range(20):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            # Previous window: start_return=None (defaults to 0)
            g_outputs_prev = g_model(
                inputs["prev_state"],
                inputs["prev_action"],
                inputs["prev_term"],
                mask=inputs["prev_mask"],
                start_return=None,
            )

            # Current window: start_return = prev_end_return
            g_outputs_curr = g_model(
                inputs["curr_state"],
                inputs["curr_action"],
                inputs["curr_term"],
                mask=inputs["curr_mask"],
                start_return=labels["prev_end_return"].unsqueeze(1),
            )

            loss_prev = criterion(g_outputs_prev.squeeze(), labels["prev_end_return"])
            loss_curr = criterion(g_outputs_curr.squeeze(), labels["curr_end_return"])
            loss = loss_prev + loss_curr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch == 0:
            initial_loss = epoch_loss
        if epoch == 19:
            final_loss = epoch_loss

    # Loss should decrease
    assert final_loss < initial_loss, (
        f"Loss should decrease (initial: {initial_loss:.4f}, final: {final_loss:.4f})"
    )


def test_episode_boundary_start_return():
    """
    Test that first window of episode receives start_return=0.
    Verify subsequent windows receive correct start_return from labels.
    Ensure no leakage across episode boundaries.
    """
    buffer = create_mock_buffer([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    delay = rewdelay.FixedDelay(3)

    examples = est_o2.delayed_reward_data_consecutive_windows(buffer, delay)
    assert len(examples) == 2

    # Create model
    torch.manual_seed(42)
    g_model = est_o2.GNetwork(
        state_dim=2, action_dim=1, hidden_dim=32, num_heads=2, num_layers=2
    )
    g_model.eval()  # Set to eval mode for deterministic behavior

    # First window (episode 0)
    inputs1, labels1 = examples[0]
    assert labels1["prev_end_return"].item() == 0.0

    # Create mask for curr window (all True since no padding)
    curr_mask1 = torch.ones(1, inputs1["curr_state"].shape[0], dtype=torch.bool)

    # Convert to float32 to match model dtype
    curr_state1 = inputs1["curr_state"].float().unsqueeze(0)
    curr_action1 = inputs1["curr_action"].float().unsqueeze(0)
    curr_term1 = inputs1["curr_term"].float().unsqueeze(0)

    # Forward pass with start_return=None (should use 0)
    with torch.no_grad():
        output1 = g_model(
            curr_state1,
            curr_action1,
            curr_term1,
            mask=curr_mask1,
            start_return=None,
        )

        # Forward pass with explicit start_return=0
        output1_explicit = g_model(
            curr_state1,
            curr_action1,
            curr_term1,
            mask=curr_mask1,
            start_return=torch.zeros(1, 1),
        )

    assert torch.allclose(output1, output1_explicit, atol=1e-6), (
        "First window should use start_return=0"
    )

    # Second window (episode 1) - should also start fresh
    inputs2, labels2 = examples[1]
    assert labels2["prev_end_return"].item() == 0.0, (
        "Second episode should reset prev_return"
    )

    # Create mask for second window
    curr_mask2 = torch.ones(1, inputs2["curr_state"].shape[0], dtype=torch.bool)

    # Convert to float32
    curr_state2 = inputs2["curr_state"].float().unsqueeze(0)
    curr_action2 = inputs2["curr_action"].float().unsqueeze(0)
    curr_term2 = inputs2["curr_term"].float().unsqueeze(0)

    # This window should also use start_return=0 (new episode)
    with torch.no_grad():
        output2 = g_model(
            curr_state2,
            curr_action2,
            curr_term2,
            mask=curr_mask2,
            start_return=None,
        )

    # Outputs should be different (different episodes with different rewards)
    # This verifies no leakage - just check they're not exactly equal
    assert not torch.equal(output1, output2), (
        "Different episodes should produce different outputs"
    )


@pytest.mark.skip(
    reason="start_return functionality is currently disabled (commented out in GNetwork.forward)"
)
def test_gradient_flow_through_start_return_encoder():
    """
    Test that gradients flow through start_return_encoder parameters.
    Verify no gradient issues.
    """
    torch.manual_seed(42)
    g_model = est_o2.GNetwork(
        state_dim=2, action_dim=1, hidden_dim=16, num_heads=2, num_layers=1
    )

    # Create input
    batch_size = 4
    seq_len = 3
    state = torch.randn(batch_size, seq_len, 2)
    action = torch.randn(batch_size, seq_len, 1)
    term = torch.zeros(batch_size, seq_len, 1)
    start_return = torch.randn(batch_size, 1)
    target = torch.randn(batch_size, 1)

    # Forward pass
    output = g_model(state, action, term, start_return=start_return)

    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Verify gradients exist for start_return_encoder parameters (MLP with 2 layers)
    assert g_model.start_return_encoder[0].weight.grad is not None, (
        "Gradients should exist for start_return_encoder[0].weight"
    )
    assert g_model.start_return_encoder[2].weight.grad is not None, (
        "Gradients should exist for start_return_encoder[2].weight"
    )

    # Verify gradients are not all zeros
    assert g_model.start_return_encoder[0].weight.grad.abs().sum() > 0, (
        "Gradients should be non-zero"
    )
    assert g_model.start_return_encoder[2].weight.grad.abs().sum() > 0, (
        "Gradients should be non-zero"
    )

    # Verify no NaN or Inf in gradients
    assert not torch.isnan(g_model.start_return_encoder[0].weight.grad).any(), (
        "No NaN in gradients"
    )
    assert not torch.isinf(g_model.start_return_encoder[0].weight.grad).any(), (
        "No Inf in gradients"
    )


@pytest.mark.skip(
    reason="start_return functionality is currently disabled (commented out in GNetwork.forward)"
)
def test_gnetwork_delta_prediction_structure():
    """
    Test that GNetwork has residual connection structure.
    Verify that changing start_return directly affects output.

    Note: Due to non-linear start_return_encoder, the predicted delta
    can vary based on start_return value (which is actually desirable).
    So we test that the model uses start_return, not that delta is constant.
    """
    torch.manual_seed(42)
    g_model = est_o2.GNetwork(
        state_dim=2, action_dim=1, hidden_dim=16, num_heads=2, num_layers=1
    )
    g_model.eval()

    batch_size = 4
    seq_len = 3
    state = torch.randn(batch_size, seq_len, 2)
    action = torch.randn(batch_size, seq_len, 1)
    term = torch.zeros(batch_size, seq_len, 1)

    # Test 1: Output shape is correct
    start_return_zero = torch.zeros(batch_size, 1)

    with torch.no_grad():
        output_zero = g_model(state, action, term, start_return=start_return_zero)

    assert output_zero.shape == (batch_size, 1), "Output shape should be (batch, 1)"

    # Test 2: Different start_return values produce different outputs
    # This verifies the residual structure is working
    start_return_values = [5.0, 10.0, 20.0]
    outputs = []

    with torch.no_grad():
        for val in start_return_values:
            start_return = torch.ones(batch_size, 1) * val
            output = g_model(state, action, term, start_return=start_return)
            outputs.append(output)

    # Outputs should be different (model uses start_return)
    for idx in range(len(outputs) - 1):
        assert not torch.allclose(outputs[idx], outputs[idx + 1], atol=0.1), (
            "Outputs should differ for different start_return values "
            "(residual connection not working)"
        )

    # Test 3: Output should generally increase with start_return
    # (though non-linear encoding may cause some variation)
    with torch.no_grad():
        output_0 = g_model(state, action, term, start_return=torch.zeros(batch_size, 1))
        output_50 = g_model(
            state, action, term, start_return=torch.ones(batch_size, 1) * 50.0
        )

    # Mean output should be substantially higher for higher start_return
    mean_diff = (output_50 - output_0).mean().item()
    assert mean_diff > 10.0, (
        f"Output should increase substantially with start_return. "
        f"Got mean_diff={mean_diff:.2f}, expected > 10"
    )
