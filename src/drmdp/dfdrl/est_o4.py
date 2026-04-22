"""
Reward prediction with learned input mask for delayed, aggregate, and anonymous feedback.

This module implements est_o4: est_o2 extended with a global binary mask over the
(state, action) input space. The mask learns which dimensions actually drive reward,
inspired by the causal reward masks C^{s→r} and C^{a→r} from the GRD paper
(Zhang et al., NeurIPS 2023).

The mask weight vector w ∈ R^D (D = state_dim + action_dim) is shared across all
time steps and examples. Three differentiable relaxations are supported during training:

  "sigmoid"  — soft gate sigmoid(w) ∈ (0,1); hard threshold float(w > 0) at inference
  "ste"      — straight-through estimator: hard binary in the forward pass, sigmoid
               gradient in the backward pass
  "gumbel"   — F.gumbel_softmax with hard=True: adds Gumbel noise for stochastic
               exploration, with STE gradient; hard threshold at inference (default)

The total loss is:
  L_total = L_reward + regu_lam * L_regu + mask_lam * mean(softmax(φ)[:, 1])

where the sparsity term mean(p_active) ∈ (0, 1) pushes activation probabilities
toward zero, masking out irrelevant dimensions. This is the bounded analogue of
GRD's cross-entropy regulariser (log P_active), scaled to work with mask_lam ≈ 0.1.
L_reward / L_regu are the aggregate-reward MSE and return-consistency terms from
est_o2.

Usage Example:
    python src/drmdp/dfdrl/est_o4.py \\
        --env MountainCarContinuous-v0 \\
        --delay 3 --buffer-num-steps 500 \\
        --mask-type gumbel --mask-lam 0.1 \\
        --local-eager-mode
"""

import argparse
import ast
import collections
import dataclasses
import io
import json
import logging
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import ray
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data, tensorboard

from drmdp import dataproc, ray_utils, rewdelay

# Spec version identifier
SPEC = "o4"


@dataclasses.dataclass(frozen=True)
class TrainingArgs:
    """Arguments for training reward prediction models.

    Attributes:
        model_type: Model architecture identifier ("mlp").
        env: Gymnasium environment name.
        max_episode_steps: Maximum steps per episode before truncation.
        delay: Mean reward delay (Poisson parameter).
        train_epochs: Number of training epochs.
        buffer_num_steps: Number of environment steps to collect per run.
        batch_size: Mini-batch size for training and evaluation.
        eval_steps: Max batches per evaluation pass.
        log_episode_frequency: Evaluate every N epochs.
        output_dir: Directory for model checkpoints and metrics.
        num_runs: Number of independent runs (one seed per run).
        regu_lam: Weight for the return-consistency regularization term (λ ∈ [0, 1]).
        mask_lam: Weight for the sparsity regularization on the input mask (≥ 0).
        mask_type: Mask relaxation strategy during training ("sigmoid", "ste", "gumbel").
        local_eager_mode: If True, run experiments in-process; otherwise submit via Ray.
        seed: Random seed for reproducibility. None uses a default.
    """

    model_type: str
    env: str
    max_episode_steps: int
    delay: int
    train_epochs: int
    buffer_num_steps: int
    batch_size: int
    eval_steps: int
    log_episode_frequency: int
    output_dir: str
    num_runs: int
    regu_lam: float
    mask_lam: float
    mask_type: str
    local_eager_mode: bool
    reward_model_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    seed: Optional[int] = None


class InputMask(nn.Module):
    """Global binary mask over (state, action, term) input dimensions.

    A learnable parameter matrix φ ∈ R^{D×2} determines which of the D = state_dim +
    action_dim + 1 input dimensions are passed to the reward network. For each dimension
    i, φ_i = [φ_i^{inactive}, φ_i^{active}] are free logits parameterising a Bernoulli
    distribution via softmax. This matches the GRD paper (Zhang et al., NeurIPS 2023)
    where φ^{s→r}_{cau} ∈ R^{|s|×2} and φ^{a→r}_{cau} ∈ R^{|a|×2}. The mask is
    shared across all time steps and examples (global, not input-dependent).

    Three relaxations are available to keep the mask differentiable during training:

      "sigmoid" — mask_i = softmax(φ_i)[active] ∈ (0, 1).
          Smooth gradients everywhere. Trains on soft-gated inputs; at inference the
          mask is decided by argmax(φ_i).

      "ste" — forward: mask_i = argmax(φ_i).
          Backward: gradient flows through softmax(φ_i)[active] (straight-through
          estimator). Truly binary in the forward pass with no train/inference gap.

      "gumbel" — F.gumbel_softmax with hard=True on φ directly.
          Adds Gumbel noise during training for stochastic exploration of the mask
          configuration space; uses STE gradient so the mask is binary in the forward
          pass. At inference: argmax(φ_i). (Default — recommended.)

    Sparsity is encouraged externally via mean(softmax(φ)[:, 1]) ∈ (0, 1) —
    minimising this pushes p_active toward zero. GRD's log P_active would give the
    same gradient direction but is unbounded below, requiring λ ~ 1e-5 to 1e-8;
    the bounded form is compatible with mask_lam ≈ 0.1.
    """

    def __init__(
        self, input_dim: int, mask_type: str = "gumbel", temperature: float = 1.0
    ):
        """Initialise mask with all dimensions active (active logit > inactive logit).

        We initialise the mask with random numbers.

        Args:
            input_dim: Number of dimensions to mask (state_dim + action_dim + 1).
            mask_type: Relaxation strategy: "sigmoid", "ste", or "gumbel".
            temperature: Gumbel-Softmax temperature τ (only used when mask_type="gumbel").
                Lower values produce sharper samples; τ=1 is the standard setting.
        """
        super().__init__()
        self.mask_type = mask_type
        self.temperature = temperature
        logits_init = torch.rand(input_dim, 2)
        self.logits = nn.Parameter(logits_init)  # (D, 2): [inactive, active]

    def forward(self) -> torch.Tensor:
        """Return the current mask of shape (input_dim,).

        Returns a soft approximation during training and a hard binary mask at inference
        (except for sigmoid, which always returns soft values).
        """
        if self.mask_type == "sigmoid":
            return F.softmax(self.logits, dim=-1)[:, 1]
        if self.mask_type == "ste":
            hard = self.logits.argmax(dim=-1).float()
            soft = F.softmax(self.logits, dim=-1)[:, 1]
            # STE: hard in the forward pass, softmax gradient in the backward pass
            return soft + (hard - soft).detach()
        if self.mask_type == "gumbel":
            if self.training:
                # φ is already (D, 2) — apply Gumbel-Softmax directly with STE gradient.
                samples = F.gumbel_softmax(
                    self.logits, tau=self.temperature, hard=True, dim=-1
                )
                return samples[:, 1]  # "active" column, shape (input_dim,)
            return self.logits.argmax(dim=-1).float()
        raise ValueError(f"Unknown mask_type: {self.mask_type!r}")

    def binary_mask(self) -> torch.Tensor:
        """Return the hard 0/1 mask via greedy argmax for logging and inference."""
        return self.logits.argmax(dim=-1).float()


class RNetwork(nn.Module):
    """Feedforward MLP for reward prediction with a learned input mask.

    The mask gates all (state, action, term) dimensions before the MLP.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        powers: int = 1,
        num_hidden_layers: int = 4,
        hidden_dim: int = 256,
        mask_type: str = "gumbel",
    ):
        """Initialise layers and the input mask.

        Args:
            state_dim: Dimensionality of the observation space.
            action_dim: Dimensionality of the action space.
            powers: Polynomial feature expansion degree.
            num_hidden_layers: Number of hidden (Linear + ReLU) pairs.
            hidden_dim: Width of each hidden layer.
            mask_type: Relaxation strategy for the input mask.
        """
        super().__init__()
        self.register_buffer("powers", torch.tensor(range(powers)) + 1)
        self.num_hidden_layers = num_hidden_layers
        self.input_mask = InputMask(state_dim + action_dim + 1, mask_type=mask_type)
        input_dim = (state_dim + action_dim + 1) * powers
        output_dim = hidden_dim if num_hidden_layers > 0 else input_dim
        layers = []
        for _ in range(self.num_hidden_layers):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(output_dim, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, term: torch.Tensor
    ) -> torch.Tensor:
        """Predict per-step rewards for a batch of sequences.

        Args:
            state: Tensor of shape (batch_size, seq_len, state_dim)
            action: Tensor of shape (batch_size, seq_len, action_dim)
            term: Tensor of shape (batch_size, seq_len, 1)

        Returns:
            Reward predictions of shape (batch_size, seq_len, 1)
        """
        out = torch.concat([state, action, term], dim=-1)  # (batch, seq, S+A+1)
        mask = self.input_mask()
        out = out * mask  # broadcast mask over (S+A+1,)
        out = torch.pow(torch.unsqueeze(out, -1), self.powers)
        out = torch.flatten(out, start_dim=2)
        out = self.layers(out)
        return self.final_layer(out)


class DictDataset(data.Dataset):
    """Dataset that stores raw (uncollated) examples with variable-length sequences."""

    def __init__(self, inputs: Sequence, labels: Sequence):
        """
        Args:
            inputs: Sequence of input dicts (one per example)
            labels: Sequence of label dicts (one per example)
        """
        self.inputs = inputs
        self.labels = labels
        self.length = len(labels)

    def __len__(self) -> int:
        """Return number of examples in the dataset."""
        return self.length

    def __getitem__(self, idx: int):
        """Return example at the given index."""
        return self.inputs[idx], self.labels[idx]


def collate_variable_length_sequences(batch):
    """Custom collate function for batching variable-length sequences.

    Pads sequences to the maximum length within the batch.

    Args:
        batch: List of (inputs_dict, labels_dict) tuples

    Returns:
        (batched_inputs_dict, batched_labels_dict) with padded sequences
    """
    inputs_list, labels_list = zip(*batch)

    seq_lengths = [inputs["state"].shape[0] for inputs in inputs_list]
    max_seq_len = max(seq_lengths)

    batched_inputs = {}
    for key in inputs_list[0].keys():
        sequences = [inputs[key] for inputs in inputs_list]
        padded_sequences = []
        for seq in sequences:
            seq_len = seq.shape[0]
            if seq_len < max_seq_len:
                pad_shape = (max_seq_len - seq_len,) + seq.shape[1:]
                padding = torch.zeros(pad_shape, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        batched_inputs[key] = torch.stack(padded_sequences)

    batched_labels = {
        "aggregate_reward": torch.stack(
            [labels["aggregate_reward"] for labels in labels_list]
        ),
        "start_return": torch.stack([labels["start_return"] for labels in labels_list]),
        "end_return": torch.stack([labels["end_return"] for labels in labels_list]),
    }

    per_step_rewards_list = [labels["per_step_rewards"] for labels in labels_list]
    padded_per_step_rewards = []
    for rewards in per_step_rewards_list:
        seq_len = rewards.shape[0]
        if seq_len < max_seq_len:
            padding = torch.zeros(max_seq_len - seq_len, dtype=rewards.dtype)
            padded_rewards = torch.cat([rewards, padding], dim=0)
        else:
            padded_rewards = rewards
        padded_per_step_rewards.append(padded_rewards)

    batched_labels["per_step_rewards"] = torch.stack(padded_per_step_rewards)

    return batched_inputs, batched_labels


def create_training_buffer(
    env: gym.Env,
    delay: rewdelay.RewardDelay,
    buffer_num_steps: int,
    seed: Optional[int] = None,
):
    """Collect trajectory data and convert to delayed reward examples."""
    buffer = dataproc.collection_traj_data(
        env, steps=buffer_num_steps, include_term=True, seed=seed
    )
    return delayed_reward_data(buffer, delay=delay)


def delayed_reward_data(buffer: Sequence, delay: rewdelay.RewardDelay) -> List:
    """Create a dataset of delayed reward sequences from a trajectory buffer.

    Converts raw trajectory data into training examples where rewards are delayed
    according to the specified delay distribution. Each example consists of a
    sequence of (state, action, term) tuples with corresponding aggregate,
    per-step rewards, and cumulative return labels.

    Args:
        buffer: List of trajectory tuples (state, action, next_state, reward, term)
        delay: RewardDelay object that determines how many steps to aggregate

    Returns:
        List of (inputs_dict, labels_dict) tuples. Sequences shorter than the
        sampled delay are discarded. Windows never span episode boundaries.
    """

    def create_traj_step(state, action, reward, term):
        return {
            "state": torch.tensor(state, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "term": torch.tensor([float(term)], dtype=torch.float32),
        }, torch.tensor(reward, dtype=torch.float32)

    def create_example(
        traj_steps: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        start_return: float,
        end_return: float,
    ):
        inputs, labels = zip(*traj_steps)
        per_step_rewards = [label.item() for label in labels]
        aggregate_reward = sum(per_step_rewards)
        label_dict = {
            "aggregate_reward": torch.tensor(aggregate_reward),
            "per_step_rewards": torch.tensor(per_step_rewards),
            "start_return": torch.tensor(start_return, dtype=torch.float32),
            "end_return": torch.tensor(end_return, dtype=torch.float32),
        }
        return data.default_collate(inputs), label_dict

    if len(buffer) == 0:
        return []

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

    cumulative_returns = np.zeros(n_steps, dtype=np.float32)
    cumulative_sum = 0.0
    for step_idx in range(n_steps):
        cumulative_sum += reward[step_idx]
        cumulative_returns[step_idx] = cumulative_sum
        # Reset at episode boundaries to prevent return carryover across episodes
        if term[step_idx]:
            cumulative_sum = 0.0

    examples = []
    idx = 0
    while idx < n_steps:
        example_steps = []
        steps = 0
        reward_delay = delay.sample()
        window_first_idx = idx
        while idx < n_steps and steps < reward_delay:
            traj_step = create_traj_step(
                states[idx][:obs_dim], action[idx], reward[idx], term[idx]
            )
            example_steps.append(traj_step)
            current_is_terminal = term[idx]
            idx += 1
            steps += 1
            if current_is_terminal:
                break
        if steps == reward_delay:
            window_last_idx = idx - 1
            if window_first_idx == 0:
                start_return = 0.0
            elif window_first_idx > 0 and term[window_first_idx - 1]:
                start_return = 0.0
            else:
                start_return = cumulative_returns[window_first_idx - 1]
            end_return = cumulative_returns[window_last_idx]
            examples.append(create_example(example_steps, start_return, end_return))
    return examples


def evaluate_model(
    model: nn.Module,
    test_ds: data.Dataset,
    batch_size: int,
    regu_lam: float,
    collect_predictions: bool = True,
    max_batches: Optional[int] = None,
    shuffle: bool = False,
) -> Tuple[Mapping[str, Sequence], List[Any]]:
    """Evaluate model using Markovian predictions.

    For each sequence, predicts reward for each (state, action, term) tuple
    independently, then sums predictions to compare with aggregate reward.

    Args:
        model: Trained model
        test_ds: Test dataset
        batch_size: Batch size for evaluation
        regu_lam: Weight applied to the return-consistency regularization MSE when
            computing the total loss.
        collect_predictions: Whether to collect detailed predictions for analysis.
        max_batches: Maximum number of batches to evaluate. If None, evaluates all.
        shuffle: Whether to shuffle the test dataloader

    Returns:
        Tuple of (metrics, predictions_list) where metrics is a dict with keys
        "reward", "regu", and "total" mapping to arrays of per-batch MSE values.
        If collect_predictions=False, predictions_list will be empty.
    """
    test_dataloader = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_length_sequences,
    )
    eval_criterion = nn.MSELoss()
    regu_criterion = nn.MSELoss()
    errors = collections.defaultdict(list)
    predictions_list = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            batch_size_actual = inputs["state"].shape[0]
            seq_len = inputs["state"].shape[1]

            per_step_predictions = []
            for step_idx in range(seq_len):
                state_t = inputs["state"][:, step_idx, :]
                action_t = inputs["action"][:, step_idx, :]
                term_t = inputs["term"][:, step_idx, :]
                state_seq = state_t.unsqueeze(1)
                action_seq = action_t.unsqueeze(1)
                term_seq = term_t.unsqueeze(1)
                reward_t = model(state_seq, action_seq, term_seq)
                if reward_t.dim() == 3:
                    reward_t = reward_t.squeeze(1)
                per_step_predictions.append(reward_t)

            predictions = torch.stack(per_step_predictions, dim=1)
            pred_window_reward = torch.sum(predictions, dim=1).squeeze(-1)
            aggregate_rewards = labels["aggregate_reward"].float()
            reward_mse = eval_criterion(pred_window_reward, aggregate_rewards)
            regu_mse = regu_criterion(
                labels["start_return"] + pred_window_reward, labels["end_return"]
            )
            mean_squared_error = reward_mse + (regu_lam * regu_mse)
            errors["reward"].append(reward_mse)
            errors["regu"].append(regu_mse)
            errors["total"].append(mean_squared_error)

            if collect_predictions:
                for batch_idx in range(batch_size_actual):
                    per_step_rewards = (
                        labels["per_step_rewards"][batch_idx].cpu().numpy()
                    )
                    predictions_list.append(
                        {
                            "state": inputs["state"][batch_idx].cpu().numpy(),
                            "action": inputs["action"][batch_idx].cpu().numpy(),
                            "term": inputs["term"][batch_idx].cpu().numpy(),
                            "actual_reward": aggregate_rewards[batch_idx].item(),
                            "per_step_rewards": per_step_rewards,
                            "predicted_reward": pred_window_reward[batch_idx].item(),
                            "per_step_predictions": predictions[batch_idx]
                            .squeeze(-1)
                            .cpu()
                            .numpy(),
                        }
                    )

            if max_batches is not None and idx + 1 >= max_batches:
                break

    metrics = {
        key: torch.stack(errors[key]).cpu().numpy()
        for key in ("reward", "regu", "total")
    }
    return metrics, predictions_list


def save_config_and_metrics(
    output_dir: str,
    model_type: str,
    env: gym.Env,
    batch_size: int,
    eval_steps: int,
    train_losses: Sequence,
    eval_losses: Mapping[str, Sequence],
    final_mse: Mapping[str, float],
    final_rmse: Mapping[str, float],
    mask_type: str = "gumbel",
    reward_model_kwargs: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Save configuration and training metrics to JSON files.

    Args:
        output_dir: Directory to save files
        model_type: Type of model used
        env: Gymnasium environment
        batch_size: Batch size used in training
        eval_steps: Number of evaluation steps
        train_losses: List of training losses per epoch
        eval_losses: Dict mapping "reward", "regu", "total" to lists of per-eval-run mean losses
        final_mse: Final mean squared error on test set
        final_rmse: Final root mean squared error on test set
        mask_type: Mask relaxation strategy used during training
        reward_model_kwargs: Extra kwargs forwarded to RNetwork

    Returns:
        hparams dict suitable for TensorBoard add_hparams
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    metrics_file = os.path.join(output_dir, f"metrics_{model_type}.json")
    with tf.io.gfile.GFile(metrics_file, "w") as writable:
        json.dump(
            {
                "model_type": model_type,
                "train_losses": train_losses,
                "eval_losses": eval_losses,
                "final_mse": final_mse,
                "final_rmse": final_rmse,
            },
            writable,
            indent=2,
        )
    logging.info("Training metrics saved to %s", metrics_file)

    config_file = os.path.join(output_dir, "config.json")
    config = {
        "spec": SPEC,
        "model_type": model_type,
        "env_name": env.spec.id if hasattr(env, "spec") and env.spec else "unknown",
        "state_dim": obs_dim,
        "action_dim": act_dim,
        "batch_size": batch_size,
        "eval_steps": eval_steps,
        "mask_type": mask_type,
        "reward_model_kwargs": reward_model_kwargs or {},
    }
    with tf.io.gfile.GFile(config_file, "w") as writable:
        json.dump(config, writable, indent=2)
    logging.info("Config saved to %s", config_file)

    hparams = {
        key: value for key, value in config.items() if key != "reward_model_kwargs"
    }
    for key, value in (reward_model_kwargs or {}).items():
        hparams[f"reward_model_kwargs.{key}"] = value
    return hparams


def _log_final_mask(model: nn.Module) -> None:
    """Log final mask density and active dimension count to INFO."""
    with torch.no_grad():
        final_mask = model.input_mask.binary_mask().cpu().numpy()
    active_dims = int(final_mask.sum())
    total_dims = len(final_mask)
    logging.info(
        "Final mask: %d/%d dimensions active (density %.2f): %s",
        active_dims,
        total_dims,
        active_dims / total_dims,
        final_mask,
    )


def train(
    env: gym.Env,
    dataset: data.Dataset,
    train_epochs: int,
    batch_size: int,
    eval_steps: int,
    log_episode_frequency: int,
    regu_lam: float,
    mask_lam: float = 0.1,
    mask_type: str = "gumbel",
    reward_model_kwargs: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    model_type: str = "mlp",
    output_dir: str = "outputs",
    on_batch_end: Optional[Callable[[Mapping[str, float]], None]] = None,
) -> Tuple[Mapping[str, float], List[Any]]:
    """Train a reward prediction model with a learned input mask.

    The mask is trained jointly with the reward network. An L1 sparsity penalty
    on mean(sigmoid(w)) encourages the mask to zero out irrelevant dimensions.
    Mask density (fraction of active dimensions) and logit histograms are logged
    to TensorBoard at the end of each epoch.

    Args:
        env: Gymnasium environment.
        dataset: Training dataset of (inputs, labels) examples.
        train_epochs: Number of full passes over the training split.
        batch_size: Mini-batch size for training and evaluation.
        eval_steps: Max batches per mid-training evaluation pass.
        log_episode_frequency: Run evaluation every this many epochs.
        regu_lam: Weight for the return-consistency regularization term (λ ∈ [0, 1]).
        mask_lam: Weight for the sparsity regularization on the mask (≥ 0).
        mask_type: Mask relaxation strategy ("sigmoid", "ste", or "gumbel").
        reward_model_kwargs: Extra kwargs forwarded to RNetwork.
        seed: Random seed for weight initialisation and data splitting.
        model_type: Model architecture to use ("mlp").
        output_dir: Directory for model checkpoints, metrics, and predictions.
        on_batch_end: Optional callback invoked after every batch with a dict
            containing {"reward", "regu", "sparsity", "total"} loss values.

    Returns:
        Tuple of (final_mse, predictions_list).
    """
    logging.info("Training with seed: %s", seed)
    torch.manual_seed(seed if seed is not None else 127)
    train_ds, test_ds = data.random_split(dataset, lengths=[0.7, 0.3])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if model_type == "mlp":
        model = RNetwork(
            state_dim=obs_dim,
            action_dim=act_dim,
            mask_type=mask_type,
            **(reward_model_kwargs or {}),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'mlp'.")

    logging.info("Training with %s model (mask_type=%s)", model_type.upper(), mask_type)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    regu_criterion = nn.MSELoss()

    train_losses = []
    eval_losses: Dict[str, List[float]] = collections.defaultdict(list)

    with tensorboard.SummaryWriter(log_dir=output_dir) as summary_writer:
        for epoch in range(train_epochs):
            train_dataloader = data.DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_variable_length_sequences,
            )
            epoch_losses = collections.defaultdict(list)
            for inputs, labels in train_dataloader:
                outputs = model(**inputs)  # (batch, seq, 1)
                window_reward = torch.sum(outputs, dim=1).squeeze(-1)
                aggregate_rewards = labels["aggregate_reward"].float()

                reward_loss = criterion(window_reward, aggregate_rewards)
                regu_loss = regu_criterion(
                    labels["start_return"] + window_reward, labels["end_return"]
                )
                # Sparsity: mean(p_active) ∈ (0,1) penalises active dimensions
                p_active = F.softmax(model.input_mask.logits, dim=-1)[:, 1]
                sparsity_loss = p_active.mean()
                loss = reward_loss + (regu_lam * regu_loss) + (mask_lam * sparsity_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses["reward"].append(reward_loss.item())
                epoch_losses["regu"].append(regu_loss.item())
                epoch_losses["sparsity"].append(sparsity_loss.item())
                epoch_losses["total"].append(loss.item())
                if on_batch_end is not None:
                    on_batch_end(
                        {
                            "reward": reward_loss.item(),
                            "regu": regu_loss.item(),
                            "sparsity": sparsity_loss.item(),
                            "total": loss.item(),
                        }
                    )

            avg_train_loss = np.mean(epoch_losses["total"]).item()
            train_losses.append(avg_train_loss)
            for key in ("reward", "regu", "sparsity", "total"):
                summary_writer.add_scalar(
                    f"MSE/train/{key}", np.mean(epoch_losses[key]), global_step=epoch
                )
                summary_writer.add_scalar(
                    f"RMSE/train/{key}",
                    np.mean(np.sqrt(epoch_losses[key])),
                    global_step=epoch,
                )

            # Log mask state once per epoch
            with torch.no_grad():
                mask_density = model.input_mask.binary_mask().mean().item()
                summary_writer.add_scalar(
                    "mask/density", mask_density, global_step=epoch
                )
                summary_writer.add_histogram(
                    "mask/logits",
                    model.input_mask.logits.detach(),
                    global_step=epoch,
                )

            if (epoch + 1) % log_episode_frequency == 0:
                eval_mse, _ = evaluate_model(
                    model,
                    test_ds,
                    batch_size=batch_size,
                    regu_lam=regu_lam,
                    collect_predictions=False,
                    max_batches=eval_steps,
                    shuffle=True,
                )
                for key in ("reward", "regu", "total"):
                    eval_losses[key].append(np.mean(eval_mse[key]).item())
                for key in ("reward", "regu", "total"):
                    summary_writer.add_scalar(
                        f"MSE/eval/{key}", np.mean(eval_losses[key]), global_step=epoch
                    )
                    summary_writer.add_scalar(
                        f"RMSE/eval/{key}",
                        np.mean(np.sqrt(eval_losses[key])),
                        global_step=epoch,
                    )
                train_rmse = np.sqrt(avg_train_loss)
                eval_rmse = np.mean(np.sqrt(eval_losses["total"]))
                logging.info(
                    "Epoch [%d/%d], Train RMSE: %.8f, Eval RMSE: %.8f, Mask density: %.2f",
                    epoch + 1,
                    train_epochs,
                    train_rmse,
                    eval_rmse,
                    mask_density,
                )

        logging.info("Final evaluation...")
        final_eval_metrics, predictions_list = evaluate_model(
            model,
            test_ds,
            batch_size,
            regu_lam=regu_lam,
        )
        final_mse = {
            key: np.mean(value).item() for key, value in final_eval_metrics.items()
        }
        final_rmse = {
            key: np.mean(np.sqrt(value)).item()
            for key, value in final_eval_metrics.items()
        }
        logging.info("Final Test RMSE: %.8f", final_rmse["total"])
        _log_final_mask(model)

        predictions_file = os.path.join(output_dir, f"predictions_{model_type}.json")
        with tf.io.gfile.GFile(predictions_file, "w") as writable:
            json.dump(
                {
                    "model_type": model_type,
                    "final_mse": final_mse,
                    "final_rmse": final_rmse,
                    "num_predictions": len(predictions_list),
                    "predictions": [
                        {
                            "state": pred["state"].tolist(),
                            "action": pred["action"].tolist(),
                            "term": pred["term"].tolist(),
                            "actual_reward": pred["actual_reward"],
                            "predicted_reward": pred["predicted_reward"],
                            "per_step_rewards": pred["per_step_rewards"].tolist(),
                            "per_step_predictions": pred[
                                "per_step_predictions"
                            ].tolist(),
                        }
                        for pred in predictions_list
                    ],
                },
                writable,
                indent=2,
            )
        logging.info("Predictions saved to %s", predictions_file)

        model_file = os.path.join(output_dir, f"model_{model_type}.pt")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        with tf.io.gfile.GFile(model_file, "wb") as writable:
            writable.write(buffer.read())
        logging.info("Model saved to %s", model_file)

        hparams = save_config_and_metrics(
            output_dir=output_dir,
            model_type=model_type,
            env=env,
            batch_size=batch_size,
            eval_steps=eval_steps,
            train_losses=train_losses,
            eval_losses=dict(eval_losses),
            final_mse=final_mse,
            final_rmse=final_rmse,
            mask_type=mask_type,
            reward_model_kwargs=reward_model_kwargs,
        )

        summary_writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={
                **{f"mse.{key}": value for key, value in final_mse.items()},
                **{f"rmse.{key}": value for key, value in final_rmse.items()},
            },
        )
        sample_input = (
            torch.zeros(1, 1, obs_dim),
            torch.zeros(1, 1, act_dim),
            torch.zeros(1, 1, 1),
        )
        summary_writer.add_graph(model, sample_input)

    return final_mse, predictions_list


def experiment(args: TrainingArgs) -> None:
    """Run a full experiment: collect data, create dataset, and train a model.

    Args:
        args: Training configuration and hyperparameters.
    """
    logging.basicConfig(level=logging.INFO)
    env = gym.make(args.env, max_episode_steps=args.max_episode_steps)
    logging.info("Spec: %s", args)

    delay = rewdelay.ClippedPoissonDelay(args.delay, min_delay=2)
    _, max_delay = delay.range()
    logging.info(
        "Collecting %d steps with delay=%d...",
        args.buffer_num_steps * max_delay,
        args.delay,
    )
    training_buffer = create_training_buffer(
        env,
        delay=delay,
        buffer_num_steps=args.buffer_num_steps * max_delay,
        seed=args.seed,
    )
    logging.info("Created %d training examples", len(training_buffer))

    inputs, labels = zip(*training_buffer)
    dataset = DictDataset(inputs=list(inputs), labels=list(labels))

    if args.model_type == "mlp":
        logging.info("=" * 80)
        logging.info("Training MLP")
        train(
            env,
            dataset=dataset,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            eval_steps=args.eval_steps,
            model_type=args.model_type,
            regu_lam=args.regu_lam,
            mask_lam=args.mask_lam,
            mask_type=args.mask_type,
            reward_model_kwargs=args.reward_model_kwargs,
            log_episode_frequency=args.log_episode_frequency,
            output_dir=args.output_dir,
            seed=args.seed,
        )


@ray.remote
def run_fn(args: TrainingArgs) -> None:
    """Wrapper for experiment() that allows Ray to report failures."""
    try:
        experiment(args)
    except Exception as err:
        logging.error("Error in task %s: %s", args.seed, err)
        sys.exit(1)


def _parse_kwargs(pairs: Optional[List[str]]) -> Dict[str, Any]:
    if not pairs:
        return {}
    result: Dict[str, Any] = {}
    for pair in pairs:
        key, _, raw_value = pair.partition("=")
        try:
            result[key] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            result[key] = raw_value
    return result


def parse_args() -> TrainingArgs:
    """Parse command-line arguments into a TrainingArgs instance."""
    parser = argparse.ArgumentParser(
        description="Train O4 reward prediction models with learned input mask"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp",
        choices=["mlp"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MountainCarContinuous-v0",
        help="Gymnasium environment",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Fixed delay for reward feedback",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=100,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--buffer-num-steps",
        type=int,
        default=100,
        help="Number of steps to collect",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20,
        help="Number of evaluation steps",
    )
    parser.add_argument(
        "--log-episode-frequency",
        type=int,
        default=5,
        help="Evaluate every N epochs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=tempfile.gettempdir(),
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs — each with its own seed",
    )
    parser.add_argument(
        "--regu-lam",
        type=float,
        default=1.0,
        help="Weight of return-consistency regularisation (λ ∈ [0, 1])",
    )
    parser.add_argument(
        "--mask-lam",
        type=float,
        default=0.01,
        help="Weight of mask sparsity regularisation (≥ 0)",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="gumbel",
        choices=["sigmoid", "ste", "gumbel"],
        help="Mask relaxation strategy during training",
    )
    parser.add_argument("--local-eager-mode", action="store_true", default=False)
    parser.add_argument(
        "--reward-model-kwargs",
        nargs="*",
        default=None,
        help="Keyword arguments for RNetwork (e.g. powers=2 hidden_dim=512)",
    )

    args, _ = parser.parse_known_args()
    arg_dict = vars(args)
    arg_dict["reward_model_kwargs"] = _parse_kwargs(arg_dict.get("reward_model_kwargs"))
    return TrainingArgs(**arg_dict)


def main() -> None:
    """Parse arguments and launch experiment runs via Ray or locally."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    ray_env: Dict[str, Any] = {}
    with ray.init(runtime_env=ray_env):
        tasks = []
        for seed in range(args.num_runs):
            seed_output_path = os.path.join(args.output_dir, str(seed))
            seed_args = dataclasses.replace(
                args, output_dir=seed_output_path, seed=seed
            )
            if args.local_eager_mode:
                experiment(seed_args)
            else:
                tasks.append(run_fn.remote(seed_args))

        ray_utils.wait_till_completion(tasks)


if __name__ == "__main__":
    main()
