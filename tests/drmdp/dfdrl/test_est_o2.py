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
    assert labels["prev_return"].item() == 0.0
    assert labels["curr_aggregate_reward"].item() == 6.0
    assert labels["curr_return"].item() == 6.0

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
    assert labels1["prev_return"].item() == 0.0
    assert labels1["curr_aggregate_reward"].item() == 6.0
    assert labels1["curr_return"].item() == 6.0

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
    assert labels2["prev_return"].item() == 6.0
    assert labels2["curr_aggregate_reward"].item() == 15.0
    assert labels2["curr_return"].item() == 21.0

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
    g_diff = labels2["curr_return"].item() - labels2["prev_return"].item()
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
    assert labels1["prev_return"].item() == 0.0
    assert labels1["curr_aggregate_reward"].item() == 6.0
    assert labels1["curr_return"].item() == 6.0

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
    assert labels2["prev_return"].item() == 0.0
    assert labels2["curr_aggregate_reward"].item() == 60.0
    assert labels2["curr_return"].item() == 60.0

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
    assert labels["prev_return"].item() == 0.0
    assert labels["curr_aggregate_reward"].item() == 6.0
    assert labels["curr_return"].item() == 6.0

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
    assert labels0_1["prev_return"].item() == 0.0
    assert labels0_1["curr_aggregate_reward"].item() == 6.0
    assert labels0_1["curr_return"].item() == 6.0
    assert inputs0_1["prev_state"].shape == (1, 2)
    assert inputs0_1["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs0_1["prev_state"], torch.zeros(1, 2))
    assert inputs0_1["prev_timestep"].item() == 0
    assert inputs0_1["curr_timestep"].item() == 2

    # ===== Episode 0, Window 2 =====
    inputs0_2, labels0_2 = examples[1]
    assert labels0_2["prev_return"].item() == 6.0
    assert labels0_2["curr_aggregate_reward"].item() == 15.0
    assert labels0_2["curr_return"].item() == 21.0
    # Should have prev from previous window
    assert torch.allclose(inputs0_2["prev_state"], inputs0_1["curr_state"])
    assert torch.allclose(inputs0_2["prev_action"], inputs0_1["curr_action"])
    assert inputs0_2["prev_timestep"].item() == 2
    assert inputs0_2["curr_timestep"].item() == 5
    # Last step is terminal (end of episode 0)
    assert inputs0_2["curr_term"][-1, 0].item() == 1.0

    # ===== Episode 2, Window 1 - MUST have zero prev (new episode) =====
    inputs2_1, labels2_1 = examples[2]
    assert labels2_1["prev_return"].item() == 0.0
    assert labels2_1["curr_aggregate_reward"].item() == 600.0
    assert labels2_1["curr_return"].item() == 600.0
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
    assert labels0["prev_return"].item() == 0.0
    assert labels0["curr_aggregate_reward"].item() == 3.0
    assert labels0["curr_return"].item() == 3.0
    assert inputs0["prev_state"].shape == (1, 2)
    assert inputs0["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs0["prev_state"], torch.zeros(1, 2))
    assert inputs0["prev_timestep"].item() == 0
    assert inputs0["curr_timestep"].item() == 2

    # ===== Episode 1 - cumulative return RESETS, NOT 9! =====
    inputs1, labels1 = examples[1]
    assert labels1["prev_return"].item() == 0.0  # Reset for new episode
    assert labels1["curr_aggregate_reward"].item() == 6.0
    assert labels1["curr_return"].item() == 6.0  # NOT 9!
    assert inputs1["prev_state"].shape == (1, 2)
    assert inputs1["curr_state"].shape == (3, 2)
    assert torch.allclose(inputs1["prev_state"], torch.zeros(1, 2))  # Reset for new episode
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
        if labels["prev_return"].item() == 0.0:
            # First window of episode - prev should be zero-filled
            if not torch.allclose(inputs["prev_state"], torch.zeros_like(inputs["prev_state"])):
                input_violations.append(f"Window {idx}: prev_state not zero-filled for first window")
            if not torch.allclose(inputs["prev_action"], torch.zeros_like(inputs["prev_action"])):
                input_violations.append(f"Window {idx}: prev_action not zero-filled for first window")
            if not torch.allclose(inputs["prev_term"], torch.zeros_like(inputs["prev_term"])):
                input_violations.append(f"Window {idx}: prev_term not zero-filled for first window")
            if inputs["prev_timestep"].item() != 0:
                input_violations.append(f"Window {idx}: prev_timestep not 0 for first window")
        else:
            # Consecutive window - prev should match previous curr
            if prev_window_inputs is not None:
                if not torch.allclose(inputs["prev_state"], prev_window_inputs["curr_state"]):
                    input_violations.append(
                        f"Window {idx}: prev_state doesn't match previous curr_state"
                    )
                if not torch.allclose(inputs["prev_action"], prev_window_inputs["curr_action"]):
                    input_violations.append(
                        f"Window {idx}: prev_action doesn't match previous curr_action"
                    )
                if not torch.allclose(inputs["prev_term"], prev_window_inputs["curr_term"]):
                    input_violations.append(
                        f"Window {idx}: prev_term doesn't match previous curr_term"
                    )

        # Store for next iteration
        prev_window_inputs = inputs

        # ===== Verify label structure =====
        required_label_keys = {"prev_return", "curr_aggregate_reward", "curr_return"}
        if set(labels.keys()) != required_label_keys:
            label_violations.append(
                f"Window {idx}: Missing or extra label keys: {labels.keys()}"
            )

        # ===== Verify regularizer assumption =====
        if labels["prev_return"].item() == 0.0:
            g_curr = labels["curr_return"].item()
            r_obs = labels["curr_aggregate_reward"].item()
            if not np.isclose(g_curr, r_obs, atol=1e-6):
                label_violations.append(f"Window {idx}: G_curr={g_curr}, R_obs={r_obs}")
            continue

        g_diff = labels["curr_return"].item() - labels["prev_return"].item()
        r_obs = labels["curr_aggregate_reward"].item()

        if not np.isclose(g_diff, r_obs, atol=1e-6):
            label_violations.append(f"Window {idx}: G_diff={g_diff}, R_obs={r_obs}")

    # Report all violations
    all_violations = label_violations + input_violations
    if all_violations:
        print("\nViolations found:")
        for violation in all_violations:
            print(f"  {violation}")
    assert len(all_violations) == 0


# =============================================================================
# Tests for est_o2.SharedStateActionEmbedding
# =============================================================================


def test_shared_embedding_normalize_flag():
    """Test that normalize flag controls behavior."""
    embed_norm = est_o2.SharedStateActionEmbedding(
        state_dim=2, action_dim=1, hidden_dim=4, normalize=True
    )
    assert hasattr(embed_norm, "state_norm")
    assert hasattr(embed_norm, "action_norm")

    embed_no_norm = est_o2.SharedStateActionEmbedding(
        state_dim=2, action_dim=1, hidden_dim=4, normalize=False
    )
    assert not hasattr(embed_no_norm, "state_norm")
    assert not hasattr(embed_no_norm, "action_norm")


def test_shared_embedding_deterministic_with_seed():
    """Test that embedding produces deterministic results with seed."""
    state = torch.randn(4, 2)
    action = torch.randn(4, 1)
    term = torch.zeros(4, 1)

    torch.manual_seed(42)
    embed1 = est_o2.SharedStateActionEmbedding(
        state_dim=2, action_dim=1, hidden_dim=8, normalize=True
    )
    output1 = embed1(state, action, term)

    torch.manual_seed(42)
    embed2 = est_o2.SharedStateActionEmbedding(
        state_dim=2, action_dim=1, hidden_dim=8, normalize=True
    )
    output2 = embed2(state, action, term)

    assert torch.allclose(output1, output2, atol=1e-6)
