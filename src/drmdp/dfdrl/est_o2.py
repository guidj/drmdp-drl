"""
Return Grounded Reward Estimation for Delayed, Aggregate, and Anonymous Feedback (DAAF).

Implements the O2 approach from specs/reward-estimation-deeprl.md:
- RNetwork: Predicts per-step rewards r̂ from (s,a) pairs
- GNetwork: Predicts episodic returns Ĝ from sequences of (s,a) pairs
- Loss function with two regularizers:
  1. ρ₁: Grounds predictions on observed aggregate feedback
  2. ρ₂: Grounds predictions on actual episodic returns

The dual prediction approach helps the model learn temporal dependencies across
consecutive windows by enforcing consistency between:
- Predicted return differences and observed aggregate rewards
- Actual return differences and predicted aggregate rewards

Key formulas (from O2 spec):
- Main loss: (R_t^o - Σ r̂(s_t,a_t))²
- ρ₁ = [(Ĝ_{t_w_i} - R^o_t) - Ĝ_{t_w_{i-1}}]²
- ρ₂ = [(G_{t_w_i} - R̂^o_t) - G_{t_w_{i-1}}]²
- Combined: L(φ) = main_loss + λ ρ₁ + ξ ρ₂

Usage Example:
    python src/drmdp/dfdrl/est_o2.py \
        --env MountainCarContinuous-v0 \
        --reward-model-type rnn \
        --delay 3 \
        --num-steps 10000 \
        --batch-size 64 \
        --lam 0.5 \
        --xi 0.5
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from drmdp import dataproc, rewdelay


class SharedStateActionEmbedding(nn.Module):
    """
    Shared embedding module for (state, action, term) tuples.
    Used by both reward and return prediction networks.

    Applies LayerNorm to state and action inputs to handle different scales
    (e.g., position vs velocity in MountainCar, or different sensor ranges).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Normalization layers for state and action (not term, which is binary)
        if normalize:
            self.state_norm = nn.LayerNorm(state_dim)
            self.action_norm = nn.LayerNorm(action_dim)

        # +1 for term flag
        self.input_proj = nn.Linear(state_dim + action_dim + 1, hidden_dim)

    def forward(self, state, action, term):
        """
        Args:
            state: Tensor of shape (..., state_dim)
            action: Tensor of shape (..., action_dim)
            term: Tensor of shape (..., 1)

        Returns:
            embeddings: Tensor of shape (..., hidden_dim)
        """
        # Apply normalization to state and action
        if self.normalize:
            state = self.state_norm(state)
            action = self.action_norm(action)

        # Concatenate and project
        features = torch.concat([state, action, term], dim=-1)
        return nn.functional.relu(self.input_proj(features))


class RNetwork(nn.Module):
    """
    Feedforward MLP for reward prediction.
    Processes (state, action, term) tuples independently.
    Uses shared embedding for state-action representation.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, shared_embedding=None):
        super().__init__()
        # Use shared embedding or create new one
        if shared_embedding is not None:
            self.embedding = shared_embedding
        else:
            self.embedding = SharedStateActionEmbedding(
                state_dim, action_dim, hidden_dim
            )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, term):
        features = self.embedding(state, action, term)
        features = nn.functional.relu(self.fc2(features))
        reward = self.fc3(features)
        return reward


class RNetworkRNN(nn.Module):
    """
    RNN-based reward prediction network.
    Processes sequential (state, action, term) tuples and outputs one reward value per step.
    Uses shared embedding for state-action representation.

    Input shape: (batch_size, sequence_length, state_dim + action_dim + 1)
    Output shape: (batch_size, sequence_length, 1)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_layers=2,
        rnn_type="lstm",
        dropout=0.1,
        shared_embedding=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # Use shared embedding or create new one
        if shared_embedding is not None:
            self.embedding = shared_embedding
        else:
            self.embedding = SharedStateActionEmbedding(
                state_dim, action_dim, hidden_dim
            )

        # RNN layer
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}. Use 'lstm' or 'gru'.")

        # Output projection: hidden_dim -> 1 (reward per step)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, term):
        """
        Args:
            state: Tensor of shape (batch_size, sequence_length, state_dim)
            action: Tensor of shape (batch_size, sequence_length, action_dim)
            term: Tensor of shape (batch_size, sequence_length, 1)

        Returns:
            rewards: Tensor of shape (batch_size, sequence_length, 1)
                Predictions for real timesteps only (stub is handled internally)
        """
        batch_size = state.shape[0]
        device = state.device

        # Prepend zero stub for position 0 to ensure consistent RNN initialization
        zero_state = torch.zeros(batch_size, 1, self.state_dim, device=device)
        zero_action = torch.zeros(batch_size, 1, self.action_dim, device=device)
        zero_term = torch.zeros(batch_size, 1, 1, device=device)

        # Create sequence with stub: [zero_stub, actual_sequence]
        state_with_stub = torch.cat([zero_state, state], dim=1)
        action_with_stub = torch.cat([zero_action, action], dim=1)
        term_with_stub = torch.cat([zero_term, term], dim=1)

        # Apply shared embedding to get hidden representation
        # Shape: (batch_size, sequence_length+1, hidden_dim)
        hidden = self.embedding(state_with_stub, action_with_stub, term_with_stub)

        # Process through RNN
        # rnn_out shape: (batch_size, sequence_length+1, hidden_dim)
        rnn_out, _ = self.rnn(hidden)

        # Project to reward predictions (one per timestep)
        # Shape: (batch_size, sequence_length+1, 1)
        rewards = self.output_proj(rnn_out)

        # Return only predictions for real timesteps (skip position 0)
        return rewards[:, 1:, :]


class RNetworkTransformer(nn.Module):
    """
    Transformer Decoder-based reward prediction network.
    Processes sequential (state, action, term) tuples and outputs one reward value per step.
    Uses shared embedding for state-action representation.

    Uses causal masking to ensure predictions at timestep t only depend on steps <= t.

    Input shape: (batch_size, sequence_length, state_dim + action_dim + 1)
    Output shape: (batch_size, sequence_length, 1)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        feedforward_dim=1024,
        dropout=0.1,
        shared_embedding=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Use shared embedding or create new one
        if shared_embedding is not None:
            self.embedding = shared_embedding
        else:
            self.embedding = SharedStateActionEmbedding(
                state_dim, action_dim, hidden_dim
            )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 1000, hidden_dim)
        )  # Max sequence length 1000

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection: hidden_dim -> 1 (reward per step)
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize positional encoding with sinusoidal pattern
        self._init_positional_encoding()

    def _init_positional_encoding(self):
        """Initialize positional encoding with sinusoidal pattern."""
        max_len = self.pos_encoding.size(1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.hidden_dim)
        )

        pe = torch.zeros(1, max_len, self.hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pe)

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for transformer decoder.
        Ensures that position i can only attend to positions <= i.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, state, action, term):
        """
        Args:
            state: Tensor of shape (batch_size, sequence_length, state_dim)
            action: Tensor of shape (batch_size, sequence_length, action_dim)
            term: Tensor of shape (batch_size, sequence_length, 1)

        Returns:
            rewards: Tensor of shape (batch_size, sequence_length, 1)
                Predictions for real timesteps only (stub is handled internally)
        """
        batch_size = state.shape[0]
        device = state.device

        # Prepend zero stub for position 0 to ensure consistent initialization
        zero_state = torch.zeros(batch_size, 1, self.state_dim, device=device)
        zero_action = torch.zeros(batch_size, 1, self.action_dim, device=device)
        zero_term = torch.zeros(batch_size, 1, 1, device=device)

        # Create sequence with stub: [zero_stub, actual_sequence]
        state_with_stub = torch.cat([zero_state, state], dim=1)
        action_with_stub = torch.cat([zero_action, action], dim=1)
        term_with_stub = torch.cat([zero_term, term], dim=1)

        seq_len = state_with_stub.shape[1]

        # Apply shared embedding to get hidden representation
        # Shape: (batch_size, sequence_length+1, hidden_dim)
        hidden = self.embedding(state_with_stub, action_with_stub, term_with_stub)

        # Add positional encoding
        hidden = hidden + self.pos_encoding[:, :seq_len, :]

        # Apply layer normalization
        hidden = self.layer_norm(hidden)

        # Generate causal mask
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(hidden.device)

        # For transformer decoder, we use the input as both memory and target
        # This is a decoder-only architecture (similar to GPT)
        # The causal mask ensures autoregressive behavior
        output = self.transformer_decoder(
            tgt=hidden,
            memory=hidden,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )

        # Project to reward predictions (one per timestep)
        # Shape: (batch_size, sequence_length+1, 1)
        rewards = self.output_proj(output)

        # Return only predictions for real timesteps (skip position 0)
        return rewards[:, 1:, :]


class GNetwork(nn.Module):
    """
    Transformer Encoder-based return prediction network.
    Processes entire previous window sequence and outputs single return value.
    Uses shared embedding for state-action representation.

    Unlike RNetworkTransformer, this uses an Encoder (not Decoder) and does not
    require causal masking since it needs to attend to the entire sequence.
    It outputs a single scalar value representing the aggregate return for the window.

    Input shape: (batch_size, sequence_length, state_dim + action_dim + 1)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        feedforward_dim=1024,
        dropout=0.1,
        shared_embedding=None,
        max_episode_steps=1000,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_episode_steps = max_episode_steps

        # Use shared embedding or create new one
        if shared_embedding is not None:
            self.embedding = shared_embedding
        else:
            self.embedding = SharedStateActionEmbedding(
                state_dim, action_dim, hidden_dim
            )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 1000, hidden_dim)
        )  # Max sequence length 1000

        # Timestep embedding: scalar timestep -> hidden_dim
        self.timestep_proj = nn.Linear(1, hidden_dim)

        # Transformer encoder layers (no causal masking needed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection: hidden_dim -> 1 (single return value)
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize positional encoding with sinusoidal pattern
        self._init_positional_encoding()

    def _init_positional_encoding(self):
        """Initialize positional encoding with sinusoidal pattern."""
        max_len = self.pos_encoding.size(1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.hidden_dim)
        )

        pe = torch.zeros(1, max_len, self.hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pe)

    def forward(self, state, action, term, mask=None, timestep=None):
        """
        Args:
            state: Tensor of shape (batch_size, sequence_length, state_dim)
            action: Tensor of shape (batch_size, sequence_length, action_dim)
            term: Tensor of shape (batch_size, sequence_length, 1)
            mask: Optional bool tensor of shape (batch_size, sequence_length)
                  True for valid positions, False for padding
            timestep: Optional tensor of shape (batch_size, 1) - absolute episode timestep t_{w_i}

        Returns:
            returns: Tensor of shape (batch_size, 1)
                Single return value per sequence
        """
        seq_len = state.shape[1]

        # Apply shared embedding to get hidden representation
        # Shape: (batch_size, sequence_length, hidden_dim)
        hidden = self.embedding(state, action, term)

        # Add positional encoding
        hidden = hidden + self.pos_encoding[:, :seq_len, :]

        # Add timestep information if provided
        if timestep is not None:
            # Normalize timestep to [0, 1] range
            norm_timestep = timestep.float() / self.max_episode_steps
            # Project to hidden dimension and add to sequence representation
            timestep_emb = self.timestep_proj(norm_timestep)  # (batch, hidden_dim)
            # Add timestep embedding to all positions in sequence
            hidden = hidden + timestep_emb.unsqueeze(1)  # Broadcast across sequence

        # Apply layer normalization
        hidden = self.layer_norm(hidden)

        # Create attention mask for transformer (inverted: True = mask out)
        # Transformer expects: False = attend, True = ignore
        if mask is not None:
            # Convert our mask (True=valid) to transformer format (True=ignore)
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        # Process through transformer encoder with masking
        # output shape: (batch_size, sequence_length, hidden_dim)
        output = self.transformer_encoder(hidden, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pooling across sequence dimension
        # Shape: (batch_size, hidden_dim)
        if mask is not None:
            # Expand mask to match hidden dimension
            # mask: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            mask_expanded = mask.unsqueeze(-1).float()
            # Zero out padding positions
            masked_output = output * mask_expanded
            # Sum and divide by actual sequence length
            pooled = masked_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = torch.mean(output, dim=1)

        # Project to single return value
        # Shape: (batch_size, 1)
        return_value = self.output_proj(pooled)

        return return_value


class DualWindowDataset(data.Dataset):
    """
    Dataset for dual-window training.
    Each example contains data from consecutive windows (previous and current).
    """

    def __init__(self, inputs: Mapping[str, Sequence], labels: Mapping[str, Sequence]):
        """
        Args:
            inputs: Dict with keys: prev_state, prev_action, prev_term,
                    curr_state, curr_action, curr_term
            labels: Dict with keys: prev_return, curr_aggregate_reward, curr_return
        """
        self.inputs = inputs
        self.labels = labels
        self.length = len(labels["prev_return"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return dictionaries for the given index
        return (
            {key: self.inputs[key][idx] for key in self.inputs},
            {key: self.labels[key][idx] for key in self.labels},
        )


def collate_variable_length_windows(batch):
    """
    Custom collate function for batching variable-length window sequences.
    Pads sequences to the maximum length in the batch and creates attention masks.

    Args:
        batch: List of (inputs_dict, labels_dict) tuples

    Returns:
        Tuple of (batched_inputs_dict, batched_labels_dict)
    """
    inputs_list, labels_list = zip(*batch)

    # Stack labels (all scalars, so no padding needed)
    batched_labels = {
        key: torch.stack([labels[key] for labels in labels_list])
        for key in labels_list[0].keys()
    }

    # Pad and batch inputs (variable-length sequences)
    batched_inputs = {}

    # Track sequence lengths for creating masks
    prev_lengths = []
    curr_lengths = []

    for key in inputs_list[0].keys():
        sequences = [inputs[key] for inputs in inputs_list]

        # Handle timestep fields separately (they are scalars, not sequences)
        if key.endswith("_timestep"):
            # Stack timestep tensors directly
            batched_inputs[key] = torch.cat(sequences, dim=0)
            continue

        # Track original lengths
        if key.startswith("prev_"):
            prev_lengths = [seq.shape[0] for seq in sequences]
        elif key.startswith("curr_"):
            curr_lengths = [seq.shape[0] for seq in sequences]

        # Get max sequence length in this batch
        max_len = max(seq.shape[0] for seq in sequences)

        # Pad sequences to max_len
        padded_sequences = []
        for seq in sequences:
            seq_len = seq.shape[0]
            if seq_len < max_len:
                # Pad with zeros
                pad_shape = (max_len - seq_len,) + seq.shape[1:]
                padding = torch.zeros(pad_shape, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Stack padded sequences
        batched_inputs[key] = torch.stack(padded_sequences)

    # Create attention masks (True = valid position, False = padding)
    batch_size = len(prev_lengths)
    prev_max_len = batched_inputs["prev_state"].shape[1]
    curr_max_len = batched_inputs["curr_state"].shape[1]

    prev_mask = torch.zeros(batch_size, prev_max_len, dtype=torch.bool)
    curr_mask = torch.zeros(batch_size, curr_max_len, dtype=torch.bool)

    for idx in range(batch_size):
        prev_mask[idx, : prev_lengths[idx]] = True
        curr_mask[idx, : curr_lengths[idx]] = True

    batched_inputs["prev_mask"] = prev_mask
    batched_inputs["curr_mask"] = curr_mask

    return batched_inputs, batched_labels


def create_training_buffer(env, delay: rewdelay.RewardDelay, num_steps: int):
    """
    Collects examples of (s,a,s',r,d) from an environment.
    """
    buffer = dataproc.collection_traj_data(env, steps=num_steps, include_term=True)
    return delayed_reward_data_consecutive_windows(buffer, delay=delay)


def delayed_reward_data_consecutive_windows(buffer, delay: rewdelay.RewardDelay):
    """
    Creates dataset with consecutive window pairs.

    For each pair of consecutive windows, creates an example with:
    - Previous window: state, action, term, cumulative_return_at_last_step
    - Current window: state, action, term, aggregate_reward, cumulative_return_at_last_step

    Cumulative return = sum of all rewards from beginning up to that step.

    For the first window, uses zero-filled previous window.

    Args:
        buffer: List of (state, action, next_state, reward, term) tuples
        delay: RewardDelay object for sampling window sizes

    Returns:
        List of (inputs_dict, labels_dict) tuples where:
        - inputs_dict: prev_state, prev_action, prev_term, curr_state, curr_action, curr_term
        - labels_dict: prev_return, curr_aggregate_reward, curr_return
    """

    def create_traj_step(state, action, reward, term):
        return {
            "state": torch.tensor(state),
            "action": torch.tensor(action),
            "term": torch.tensor([float(term)]),
        }, torch.tensor(reward, dtype=torch.float32)

    def create_window(traj_steps: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        """Create window data from trajectory steps."""
        inputs, labels = zip(*traj_steps)
        # Collate inputs and sum rewards
        return data.default_collate(inputs), torch.sum(torch.stack(labels))

    # Extract buffer data
    states = np.concatenate(
        [
            np.stack([example[0] for example in buffer]),
            np.stack([example[2] for example in buffer]),
        ],
        axis=1,
    )
    action = np.stack([example[1] for example in buffer])
    reward = np.stack([example[3] for example in buffer])
    term = np.stack([example[4] for example in buffer])
    obs_dim = states.shape[1] // 2
    n_steps = states.shape[0]

    # Compute cumulative returns (sum from beginning to each timestep)
    cumulative_returns = np.zeros(n_steps, dtype=np.float32)
    cumulative_sum = 0.0
    for step_idx in range(n_steps):
        cumulative_sum += reward[step_idx]
        cumulative_returns[step_idx] = cumulative_sum
        # Reset cumulative sum at episode boundaries
        if term[step_idx]:
            cumulative_sum = 0.0

    # Track episode timesteps (timestep within each episode)
    episode_timesteps = np.zeros(n_steps, dtype=np.int64)
    ep_t = 0
    for step_idx in range(n_steps):
        episode_timesteps[step_idx] = ep_t
        ep_t += 1
        if term[step_idx]:
            ep_t = 0

    examples = []
    idx = 0
    prev_window = None
    prev_window_timestep = 0

    while idx < n_steps:
        # Collect current window
        curr_steps = []
        steps = 0
        reward_delay = delay.sample()
        episode_boundary_in_window = False

        while steps < reward_delay and idx < n_steps:
            traj_step = create_traj_step(
                states[idx][:obs_dim], action[idx], reward[idx], term[idx]
            )
            curr_steps.append(traj_step)

            idx += 1
            steps += 1

            # Check if we hit an episode boundary (except at the last step of window)
            # If so, stop collecting immediately to avoid contaminating with next episode
            if term[idx - 1] and steps < reward_delay:
                episode_boundary_in_window = True
                break

        # Only create example if we have a complete window AND no episode boundary
        if steps == reward_delay and not episode_boundary_in_window:
            curr_window_inputs, curr_aggregate_reward = create_window(curr_steps)
            curr_window_last_idx = idx - 1
            curr_window_timestep = episode_timesteps[curr_window_last_idx]

            # For first window, create zero-filled previous window
            if prev_window is None:
                # Create minimal zero-filled previous window (1 timestep)
                zero_prev_inputs = {
                    "state": torch.zeros(1, obs_dim),
                    "action": torch.zeros(1, action.shape[1]),
                    "term": torch.zeros(1, 1),
                }
                zero_prev_return = torch.tensor(0.0, dtype=torch.float32)
                prev_window = (zero_prev_inputs, zero_prev_return)
                prev_window_timestep = 0

            prev_window_inputs, prev_return = prev_window

            # Create dual-window example
            example_inputs = {
                "prev_state": prev_window_inputs["state"],
                "prev_action": prev_window_inputs["action"],
                "prev_term": prev_window_inputs["term"],
                "prev_timestep": torch.tensor(
                    [[prev_window_timestep]], dtype=torch.long
                ),
                "curr_state": curr_window_inputs["state"],
                "curr_action": curr_window_inputs["action"],
                "curr_term": curr_window_inputs["term"],
                "curr_timestep": torch.tensor(
                    [[curr_window_timestep]], dtype=torch.long
                ),
            }
            # Get cumulative return at the LAST step of current window
            curr_window_return = torch.tensor(
                cumulative_returns[curr_window_last_idx], dtype=torch.float32
            )

            example_labels = {
                "prev_return": prev_return,
                "curr_aggregate_reward": curr_aggregate_reward,
                "curr_return": curr_window_return,
            }

            examples.append((example_inputs, example_labels))

            # Check if current window ended with a terminal state
            # If so, next window starts a new episode and needs zero-filled previous
            if term[curr_window_last_idx]:
                # Episode ended at the last step of this window
                # Reset previous window so next window gets zero-filled prev
                prev_window = None
                prev_window_timestep = 0
            else:
                # Normal case: consecutive windows within same episode
                prev_window = (curr_window_inputs, curr_window_return)
                prev_window_timestep = curr_window_timestep
        elif episode_boundary_in_window:
            # Reset previous window after episode boundary
            # Next window will start with zero-filled previous window
            prev_window = None
            prev_window_timestep = 0

    return examples


def evaluate_dual_model(
    r_model: nn.Module,
    g_model: nn.Module,
    test_ds: data.Dataset,
    batch_size: int,
    collect_predictions: bool = True,
    max_batches: Optional[int] = None,
    shuffle: bool = False,
) -> Tuple[Dict[str, float], List[Any]]:
    """
    Evaluate both reward and return models.

    For reward model: predicts each (state, action, term) independently (Markovian),
    then sums predictions per window to compare with aggregate reward.

    For return model: forward pass on entire previous window sequence,
    direct comparison to previous aggregate reward.

    Args:
        r_model: Reward prediction model
        g_model: Return prediction model
        test_ds: Test dataset
        batch_size: Batch size for evaluation
        collect_predictions: Whether to collect detailed predictions for analysis
        max_batches: Maximum number of batches to evaluate
        shuffle: Whether to shuffle the test dataloader

    Returns:
        Tuple of (metrics_dict, predictions_list)
        metrics_dict contains: reward_mse, return_mse, combined_mse
    """
    test_dataloader = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_length_windows,
    )
    eval_criterion = nn.MSELoss()
    reward_errors = []
    return_errors = []
    rho1_errors = []
    rho2_errors = []
    predictions_list = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            batch_size_actual = inputs["curr_state"].shape[0]
            curr_seq_len = inputs["curr_state"].shape[1]

            # ===== Reward model evaluation (Markovian predictions) =====
            # Predict reward for each (state, action, term) tuple independently
            per_step_predictions = []
            for step_idx in range(curr_seq_len):
                # Extract (state, action, term) at timestep step_idx for all sequences in batch
                state_t = inputs["curr_state"][:, step_idx, :]  # (batch_size, state_dim)
                action_t = inputs["curr_action"][:, step_idx, :]  # (batch_size, action_dim)
                term_t = inputs["curr_term"][:, step_idx, :]  # (batch_size, 1)

                # Add sequence dimension for RNN/Transformer models
                state_seq = state_t.unsqueeze(1)  # (batch_size, 1, state_dim)
                action_seq = action_t.unsqueeze(1)  # (batch_size, 1, action_dim)
                term_seq = term_t.unsqueeze(1)  # (batch_size, 1, 1)

                # Forward pass
                reward_t = r_model(state_seq, action_seq, term_seq)
                if reward_t.dim() == 3:
                    reward_t = reward_t.squeeze(1)  # (batch_size, 1)
                per_step_predictions.append(reward_t)

            # Stack predictions: (batch_size, seq_len, 1)
            reward_predictions = torch.stack(per_step_predictions, dim=1)

            # Sum predictions across sequence to get aggregate reward
            pred_curr_reward = torch.squeeze(torch.sum(reward_predictions, dim=1))
            reward_mse = eval_criterion(
                pred_curr_reward, labels["curr_aggregate_reward"]
            )
            reward_errors.append(reward_mse)

            # ===== Return model evaluation =====
            # Forward pass on entire previous window sequence
            return_predictions_prev = g_model(
                inputs["prev_state"],
                inputs["prev_action"],
                inputs["prev_term"],
                mask=inputs["prev_mask"],
                timestep=inputs["prev_timestep"],
            )  # Shape: (batch_size, 1)

            # Forward pass on current window (independent prediction)
            return_predictions_curr = g_model(
                inputs["curr_state"],
                inputs["curr_action"],
                inputs["curr_term"],
                mask=inputs["curr_mask"],
                timestep=inputs["curr_timestep"],
            )  # Shape: (batch_size, 1)

            pred_prev_return = torch.squeeze(return_predictions_prev)
            pred_curr_return = torch.squeeze(return_predictions_curr)
            return_mse = eval_criterion(pred_prev_return, labels["prev_return"])
            return_errors.append(return_mse)

            # ===== Compute regularizer terms =====
            # Following O2 specification with independent predictions
            r_hat = reward_predictions
            g_hat_prev = pred_prev_return
            g_hat_curr = pred_curr_return  # Independent prediction, not compositional

            R_obs_curr = labels["curr_aggregate_reward"]
            G_actual_prev = labels["prev_return"]
            G_actual_curr = labels["curr_return"]
            R_hat_curr = pred_curr_reward

            # ρ₁: [(Ĝ_curr - R_obs) - Ĝ_prev]²
            rho_1_val = torch.mean((g_hat_curr - R_obs_curr - g_hat_prev) ** 2)
            rho1_errors.append(rho_1_val)

            # ρ₂: [(G_curr - R̂_obs) - G_prev]²
            rho_2_val = torch.mean((G_actual_curr - R_hat_curr - G_actual_prev) ** 2)
            rho2_errors.append(rho_2_val)

            # Optionally collect predictions for analysis
            if collect_predictions:
                for idx in range(batch_size_actual):
                    predictions_list.append(
                        {
                            "prev_window": {
                                "state": inputs["prev_state"][idx].cpu().numpy(),
                                "action": inputs["prev_action"][idx].cpu().numpy(),
                                "term": inputs["prev_term"][idx].cpu().numpy(),
                                "actual_return": labels["prev_return"][idx].item(),
                                "predicted_return": pred_prev_return[idx].item(),
                            },
                            "curr_window": {
                                "state": inputs["curr_state"][idx].cpu().numpy(),
                                "action": inputs["curr_action"][idx].cpu().numpy(),
                                "term": inputs["curr_term"][idx].cpu().numpy(),
                                "actual_aggregate_reward": labels[
                                    "curr_aggregate_reward"
                                ][idx].item(),
                                "predicted_aggregate_reward": pred_curr_reward[
                                    idx
                                ].item(),
                                "per_step_predictions": reward_predictions[idx]
                                .squeeze(-1)
                                .cpu()
                                .numpy(),
                            },
                        }
                    )

            # Stop early if max_batches is specified
            if max_batches is not None and idx + 1 >= max_batches:
                break

    # Calculate metrics
    reward_mse = torch.mean(torch.stack(reward_errors)).item()
    return_mse = torch.mean(torch.stack(return_errors)).item()
    combined_mse = (reward_mse + return_mse) / 2.0
    rho_1 = torch.mean(torch.stack(rho1_errors)).item()
    rho_2 = torch.mean(torch.stack(rho2_errors)).item()

    metrics = {
        "reward_mse": reward_mse,
        "return_mse": return_mse,
        "combined_mse": combined_mse,
        "rho_1": rho_1,
        "rho_2": rho_2,
    }

    return metrics, predictions_list


def train(
    env: gym.Env,
    dataset: data.Dataset,
    batch_size: int,
    eval_steps: int,
    reward_model_type: str = "rnn",
    lam: float = 1.0,
    xi: float = 1.0,
    output_dir: str = "outputs",
    seed: Optional[int] = None,
):
    """
    Train dual-window reward and return prediction models with shared embeddings.

    Implements the O2 approach with regularizers ρ₁ and ρ₂ for grounded reward estimation.

    Args:
        env: Gymnasium environment
        dataset: Training dataset (DualWindowDataset)
        batch_size: Batch size for training
        eval_steps: Number of evaluation steps
        reward_model_type: Type of reward model ("mlp", "rnn", or "transformer")
        lam: Weight for ρ₁ regularizer (grounds predictions on aggregate feedback) (default: 1.0)
        xi: Weight for ρ₂ regularizer (grounds predictions on actual returns) (default: 1.0)
        output_dir: Directory to save predictions and results
        seed: Random seed for reproducibility (default: None)
    """
    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # For maximum reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Set random seed to {seed} for reproducibility")

    train_ds, test_ds = data.random_split(dataset, lengths=[0.7, 0.3])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # ===== Create shared embedding layer =====
    shared_embedding = SharedStateActionEmbedding(
        state_dim=obs_dim, action_dim=act_dim, hidden_dim=256
    )

    # ===== Reward model (user-selectable architecture) =====
    r_model: nn.Module
    if reward_model_type == "mlp":
        r_model = RNetwork(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=256,
            shared_embedding=shared_embedding,
        )
    elif reward_model_type == "rnn":
        r_model = RNetworkRNN(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=256,
            num_layers=2,
            rnn_type="lstm",
            shared_embedding=shared_embedding,
        )
    elif reward_model_type == "transformer":
        r_model = RNetworkTransformer(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=256,
            num_heads=2,
            num_layers=2,
            shared_embedding=shared_embedding,
        )
    else:
        raise ValueError(
            f"Unknown reward_model_type: {reward_model_type}. Use 'mlp', 'rnn', or 'transformer'."
        )

    # ===== Return model (always Transformer) =====
    # Get max episode steps from environment spec, default to 1000
    max_episode_steps = (
        getattr(env.spec, "max_episode_steps", 1000)
        if hasattr(env, "spec") and env.spec
        else 1000
    )

    g_model: GNetwork = GNetwork(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=256,
        num_heads=2,
        num_layers=2,
        shared_embedding=shared_embedding,
        max_episode_steps=max_episode_steps,
    )

    print(
        f"Training with shared embeddings - "
        f"Reward model: {reward_model_type.upper()}, Return model: TRANSFORMER"
    )

    # ===== Combined optimizer =====
    # Collect unique parameters to avoid duplicating shared embedding params
    seen_params = set()
    unique_params = []
    for model in [r_model, g_model]:
        for param in model.parameters():
            if id(param) not in seen_params:
                seen_params.add(id(param))
                unique_params.append(param)

    optimizer = optim.Adam(unique_params, lr=0.01)
    criterion = nn.MSELoss()

    # Training Loop
    epochs = 100
    train_main_losses = []
    train_rho1_losses = []
    train_rho2_losses = []
    train_combined_losses = []
    eval_metrics_history = []

    for epoch in range(epochs):
        train_dataloader = data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_variable_length_windows,
        )

        # Training
        epoch_main_losses = []
        epoch_rho1_losses = []
        epoch_rho2_losses = []
        epoch_combined_losses = []

        for inputs, labels in train_dataloader:
            # ===== Forward pass on current window (reward model) =====
            r_outputs = r_model(
                inputs["curr_state"], inputs["curr_action"], inputs["curr_term"]
            )  # Shape: (batch_size, curr_seq_len, 1)

            # ===== Forward pass on previous window (return model) =====
            g_outputs_prev = g_model(
                inputs["prev_state"],
                inputs["prev_action"],
                inputs["prev_term"],
                mask=inputs["prev_mask"],
                timestep=inputs["prev_timestep"],
            )  # Shape: (batch_size, 1)

            # ===== Forward pass on current window (return model) =====
            # Following O2 spec: Ĝ_{t_w_i} = h({s_t,a_t}_{w_i}, t_{w_i})
            # Both windows get independent predictions from the return model
            g_outputs_curr = g_model(
                inputs["curr_state"],
                inputs["curr_action"],
                inputs["curr_term"],
                mask=inputs["curr_mask"],
                timestep=inputs["curr_timestep"],
            )  # Shape: (batch_size, 1)

            # ===== O2 Specification Loss Function =====
            # Extract predictions and labels
            r_hat = r_outputs  # (batch, seq_len, 1) - per-step reward predictions
            g_hat_prev = torch.squeeze(
                g_outputs_prev
            )  # (batch,) - predicted return for prev window
            g_hat_curr = torch.squeeze(
                g_outputs_curr
            )  # (batch,) - predicted return for curr window (INDEPENDENT prediction)

            R_obs_curr = labels["curr_aggregate_reward"]  # Observed aggregate reward
            G_actual_prev = labels["prev_return"]  # Actual return at end of prev window
            G_actual_curr = labels["curr_return"]  # Actual return at end of curr window

            # Main loss: Match observed aggregate reward with summed predicted rewards
            R_hat_curr = torch.squeeze(torch.sum(r_hat, dim=1))  # Predicted aggregate
            loss_main = criterion(R_hat_curr, R_obs_curr)

            # ρ₁: [(Ĝ_{t_w_i} - R^o_t) - Ĝ_{t_w_{i-1}}]²
            # Difference in predicted returns should match observed aggregate
            # Ĝ_curr - Ĝ_prev should equal R_obs_curr
            rho_1 = torch.mean((g_hat_curr - R_obs_curr - g_hat_prev) ** 2)

            # ρ₂: [(G_{t_w_i} - R̂^o_t) - G_{t_w_{i-1}}]²
            # Difference in actual returns should match predicted aggregate
            # G_curr - G_prev should equal R_hat_curr
            rho_2 = torch.mean((G_actual_curr - R_hat_curr - G_actual_prev) ** 2)

            # Combined loss with two regularization weights
            # Total: L(φ) = main_loss + λ ρ₁ + ξ ρ₂
            loss = loss_main + lam * rho_1 + xi * rho_2

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_main_losses.append(loss_main.item())
            epoch_rho1_losses.append(rho_1.item())
            epoch_rho2_losses.append(rho_2.item())
            epoch_combined_losses.append(loss.item())

        avg_main_loss = np.mean(epoch_main_losses)
        avg_rho1_loss = np.mean(epoch_rho1_losses)
        avg_rho2_loss = np.mean(epoch_rho2_losses)
        avg_combined_loss = np.mean(epoch_combined_losses)

        train_main_losses.append(avg_main_loss)
        train_rho1_losses.append(avg_rho1_loss)
        train_rho2_losses.append(avg_rho2_loss)
        train_combined_losses.append(avg_combined_loss)

        # Evaluation
        if (epoch + 1) % 5 == 0:
            eval_metrics, _ = evaluate_dual_model(
                r_model,
                g_model,
                test_ds,
                batch_size=batch_size,
                collect_predictions=False,
                max_batches=eval_steps,
                shuffle=True,
            )
            eval_metrics_history.append(eval_metrics)

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {avg_combined_loss:.4f} "
                f"(Main: {avg_main_loss:.4f}, ρ₁: {avg_rho1_loss:.4f}, ρ₂: {avg_rho2_loss:.4f}) | "
                f"Eval MSE - Reward: {eval_metrics['reward_mse']:.4f}, "
                f"Return: {eval_metrics['return_mse']:.4f}, "
                f"ρ₁: {eval_metrics['rho_1']:.4f}, "
                f"ρ₂: {eval_metrics['rho_2']:.4f}"
            )

    # Final evaluation and save predictions
    print("\nFinal evaluation...")
    final_metrics, predictions_list = evaluate_dual_model(
        r_model, g_model, test_ds, batch_size
    )
    print(f"Final Test Reward RMSE: {np.sqrt(final_metrics['reward_mse']):.8f}")
    print(f"Final Test Return RMSE: {np.sqrt(final_metrics['return_mse']):.8f}")
    print(f"Final Test Combined RMSE: {np.sqrt(final_metrics['combined_mse']):.8f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_file = output_path / f"predictions_{reward_model_type}_return.json"
    with open(predictions_file, "w") as writable:
        json.dump(
            {
                "reward_model_type": reward_model_type,
                "return_model_type": "transformer",
                "final_reward_mse": final_metrics["reward_mse"],
                "final_return_mse": final_metrics["return_mse"],
                "final_combined_mse": final_metrics["combined_mse"],
                "num_predictions": len(predictions_list),
                "predictions": [
                    {
                        "prev_window": {
                            "state": pred["prev_window"]["state"].tolist(),
                            "action": pred["prev_window"]["action"].tolist(),
                            "term": pred["prev_window"]["term"].tolist(),
                            "actual_return": pred["prev_window"]["actual_return"],
                            "predicted_return": pred["prev_window"]["predicted_return"],
                        },
                        "curr_window": {
                            "state": pred["curr_window"]["state"].tolist(),
                            "action": pred["curr_window"]["action"].tolist(),
                            "term": pred["curr_window"]["term"].tolist(),
                            "actual_aggregate_reward": pred["curr_window"][
                                "actual_aggregate_reward"
                            ],
                            "predicted_aggregate_reward": pred["curr_window"][
                                "predicted_aggregate_reward"
                            ],
                            "per_step_predictions": pred["curr_window"][
                                "per_step_predictions"
                            ].tolist(),
                        },
                    }
                    for pred in predictions_list
                ],
            },
            writable,
            indent=2,
        )
    print(f"Predictions saved to {predictions_file}")

    # Save training metrics
    metrics_file = output_path / f"metrics_{reward_model_type}_return.json"
    with open(metrics_file, "w") as writable:
        json.dump(
            {
                "reward_model_type": reward_model_type,
                "return_model_type": "transformer",
                "train_main_losses": train_main_losses,
                "train_rho1_losses": train_rho1_losses,
                "train_rho2_losses": train_rho2_losses,
                "train_combined_losses": train_combined_losses,
                "eval_metrics_history": eval_metrics_history,
                "final_reward_mse": final_metrics["reward_mse"],
                "final_return_mse": final_metrics["return_mse"],
                "final_combined_mse": final_metrics["combined_mse"],
            },
            writable,
            indent=2,
        )
    print(f"Training metrics saved to {metrics_file}")

    # Save models
    model_file = output_path / f"model_{reward_model_type}_return.pt"
    torch.save(
        {
            "reward_model_state_dict": r_model.state_dict(),
            "return_model_state_dict": g_model.state_dict(),
            "reward_model_type": reward_model_type,
            "return_model_type": "transformer",
        },
        model_file,
    )
    print(f"Models saved to {model_file}")

    return final_metrics, predictions_list


def main():
    parser = argparse.ArgumentParser(
        description="Train dual-window reward and return prediction models for delayed feedback"
    )
    parser.add_argument(
        "--reward-model-type",
        type=str,
        default="mlp",
        choices=["mlp", "rnn", "transformer"],
        help="Model architecture for reward prediction (default: rnn)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MountainCarContinuous-v0",
        help="Gymnasium environment (default: MountainCarContinuous-v0)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2500,
        help="Maximum steps per episode (default: 2500)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Fixed delay for reward feedback (default: 3)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100_000,
        help="Number of steps to collect (default: 100000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20,
        help="Number of evaluation steps (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Weight for ρ₁ regularizer (return prediction grounding on aggregate feedback) (default: 1.0)",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=1.0,
        help="Weight for ρ₂ regularizer (reward prediction grounding on actual returns) (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()

    # Set random seeds for data collection if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed to {args.seed} for reproducibility")

    # Create environment
    env = gym.make(args.env, max_episode_steps=args.max_episode_steps)
    print(f"Environment: {args.env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create delay and training buffer
    delay = rewdelay.FixedDelay(args.delay)
    _, max_delay = delay.range()
    print(f"\nCollecting {args.num_steps * max_delay} steps with delay={args.delay}...")
    training_buffer = create_training_buffer(
        env, delay=delay, num_steps=args.num_steps * max_delay
    )
    print(f"Created {len(training_buffer)} training examples")

    # Create dataset
    inputs_list, labels_list = zip(*training_buffer)

    # Separate inputs and labels by window
    inputs_dict = {
        "prev_state": [inp["prev_state"] for inp in inputs_list],
        "prev_action": [inp["prev_action"] for inp in inputs_list],
        "prev_term": [inp["prev_term"] for inp in inputs_list],
        "prev_timestep": [inp["prev_timestep"] for inp in inputs_list],
        "curr_state": [inp["curr_state"] for inp in inputs_list],
        "curr_action": [inp["curr_action"] for inp in inputs_list],
        "curr_term": [inp["curr_term"] for inp in inputs_list],
        "curr_timestep": [inp["curr_timestep"] for inp in inputs_list],
    }
    labels_dict = {
        "prev_return": torch.stack([lab["prev_return"] for lab in labels_list]),
        "curr_aggregate_reward": torch.stack(
            [lab["curr_aggregate_reward"] for lab in labels_list]
        ),
        "curr_return": torch.stack([lab["curr_return"] for lab in labels_list]),
    }

    dataset = DualWindowDataset(inputs=inputs_dict, labels=labels_dict)

    # Train models
    train(
        env,
        dataset=dataset,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        reward_model_type=args.reward_model_type,
        lam=args.lam,
        xi=args.xi,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
