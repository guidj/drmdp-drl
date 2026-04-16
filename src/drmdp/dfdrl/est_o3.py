"""
EM-based reward prediction for delayed, aggregate, and anonymous feedback (DAAF).

This module implements est_o3, an Expectation-Maximisation formulation of reward
estimation from delayed aggregate feedback.

At each training iteration the loop performs two steps per batch:

  E-step (no gradient): Compute per-step soft targets
      t̃ₜ = μₜ + (Yᵥ - Σⱼ μⱼ) / D
  where μₜ = f_θ(sₜ, aₜ) are current predictions, Yᵥ is the observed aggregate
  reward, and D is the actual (unpadded) window length.

  M-step (backward pass): Minimise per-step MSE against soft targets, masked to
  exclude zero-padded positions:
      L_reward = (1 / D) · Σₜ maskₜ · (f_θ(sₜ, aₜ) - t̃ₜ)²

The return-consistency regularisation from est_o2 is preserved:
  L_regu = MSE(start_return + Σₜ maskₜ · f̂(sₜ, aₜ), end_return)

Both est_o2 and est_o3 share the same fixed points (Σₜ f̂(sₜ,aₜ) = Yᵥ); EM
additionally guarantees monotone marginal likelihood increase per iteration and
provides per-step gradient signal rather than one aggregate residual per window.
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
from torch import nn, optim
from torch.utils import data, tensorboard

from drmdp import dataproc, ray_utils, rewdelay

# Spec version identifier
SPEC = "o3"


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
    local_eager_mode: bool
    reward_model_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    seed: Optional[int] = None


class RNetwork(nn.Module):
    """
    Feedforward MLP for reward prediction.
    Processes (state, action, term) tuples independently.
    """

    def __init__(
        self, state_dim, action_dim, powers=1, num_hidden_layers=4, hidden_dim=256
    ):
        """Initialize network layers for the given state and action dimensions."""
        super().__init__()
        self.register_buffer("powers", torch.tensor(range(powers)) + 1)
        self.num_hidden_layers = num_hidden_layers
        # +1 for term flag
        input_dim = (state_dim + action_dim + 1) * powers
        output_dim = hidden_dim if num_hidden_layers > 0 else input_dim
        layers = []
        for _ in range(self.num_hidden_layers):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(output_dim, 1)

    def forward(self, state, action, term):
        """
        Args:
            state: Tensor of shape (batch_size, time step, state_dim)
            action: Tensor of shape (batch_size, time step, action_dim)
            term: Tensor of shape (batch_size, time step, 1)

        Returns:
            reward: Tensor of shape (batch_size, seq_len, 1)
        """
        out = torch.concat([state, action, term], dim=-1)
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

    def __len__(self):
        """Return number of examples in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Return example at the given index."""
        return self.inputs[idx], self.labels[idx]


def collate_variable_length_sequences(batch):
    """
    Custom collate function for batching variable-length sequences.
    Pads sequences to the maximum length within the batch.

    Args:
        batch: List of (inputs_dict, labels_dict) tuples

    Returns:
        (batched_inputs_dict, batched_labels_dict) with padded sequences.
        batched_labels_dict includes 'seq_lengths' (LongTensor of shape (batch,))
        containing the actual (unpadded) length of each sequence, required by
        the EM E-step to distribute residuals by true window length.
    """
    inputs_list, labels_list = zip(*batch)

    # Find max sequence length in this batch
    seq_lengths = [inputs["state"].shape[0] for inputs in inputs_list]
    max_seq_len = max(seq_lengths)

    # Pad and stack inputs
    batched_inputs = {}
    for key in inputs_list[0].keys():
        sequences = [inputs[key] for inputs in inputs_list]

        # Pad each sequence to max_seq_len
        padded_sequences = []
        for seq in sequences:
            seq_len = seq.shape[0]
            if seq_len < max_seq_len:
                # Pad with zeros
                pad_shape = (max_seq_len - seq_len,) + seq.shape[1:]
                padding = torch.zeros(pad_shape, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Stack: (batch_size, max_seq_len, dim)
        batched_inputs[key] = torch.stack(padded_sequences)

    # Stack labels
    batched_labels = {
        "aggregate_reward": torch.stack(
            [labels["aggregate_reward"] for labels in labels_list]
        ),
        "start_return": torch.stack([labels["start_return"] for labels in labels_list]),
        "end_return": torch.stack([labels["end_return"] for labels in labels_list]),
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.long),
    }

    # Pad per_step_rewards to match max_seq_len
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
    env, delay: rewdelay.RewardDelay, buffer_num_steps: int, seed: Optional[int] = None
):
    """
    Collects examples of (s,a,s',r,d) from an environment.
    """
    buffer = dataproc.collection_traj_data(
        env, steps=buffer_num_steps, include_term=True, seed=seed
    )
    return delayed_reward_data(buffer, delay=delay)


def delayed_reward_data(buffer, delay: rewdelay.RewardDelay):
    """
    Creates a dataset of delayed reward sequences from trajectory buffer.

    Converts raw trajectory data into training examples where rewards are delayed
    according to the specified delay distribution. Each example consists of a
    sequence of (state, action, term) tuples with corresponding aggregate and
    per-step rewards.

    Args:
        buffer: List of trajectory tuples (state, action, next_state, reward, term)
        delay: RewardDelay object that determines how many steps to aggregate

    Returns:
        List of tuples (inputs, labels) where:
            - inputs: Dict with batched tensors of 'state', 'action', 'term'
            - labels: Dict with 'aggregate_reward' (sum), 'per_step_rewards' (list),
              'start_return', 'end_return'

    Note:
        Sequences shorter than the sampled delay are discarded.
        Windows never span episode boundaries.
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
        # assumes labels single value tensors
        inputs, labels = zip(*traj_steps)

        # Extract per-step rewards and compute aggregate
        per_step_rewards = [label.item() for label in labels]
        aggregate_reward = sum(per_step_rewards)

        # Return dict with both aggregate and per-step rewards
        label_dict = {
            "aggregate_reward": torch.tensor(aggregate_reward),
            "per_step_rewards": torch.tensor(per_step_rewards),
            "start_return": torch.tensor(start_return, dtype=torch.float32),
            "end_return": torch.tensor(end_return, dtype=torch.float32),
        }

        return data.default_collate(inputs), label_dict

    # Handle empty buffer
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

    # Compute cumulative returns (sum from episode start to each timestep)
    cumulative_returns = np.zeros(n_steps, dtype=np.float32)
    cumulative_sum = 0.0
    for step_idx in range(n_steps):
        cumulative_sum += reward[step_idx]
        cumulative_returns[step_idx] = cumulative_sum
        # Reset at episode boundaries
        if term[step_idx]:
            cumulative_sum = 0.0

    examples = []
    idx = 0
    while idx < n_steps:
        example_steps = []
        steps = 0
        reward_delay = delay.sample()
        window_first_idx = idx  # Track first step index in window
        while idx < n_steps and steps < reward_delay:
            traj_step = create_traj_step(
                states[idx][:obs_dim], action[idx], reward[idx], term[idx]
            )
            example_steps.append(traj_step)
            current_is_terminal = term[idx]
            idx += 1
            steps += 1
            # Stop window at terminal state (episode boundary)
            if current_is_terminal:
                break
        # Keep window only if it has the expected delay length
        if steps == reward_delay:
            window_last_idx = idx - 1  # Last step index in window

            # Compute start_return (cumulative return before window starts)
            if window_first_idx == 0:
                start_return = 0.0  # Window starts at episode beginning
            elif window_first_idx > 0 and term[window_first_idx - 1]:
                start_return = 0.0  # Window starts at new episode (after terminal)
            else:
                start_return = cumulative_returns[window_first_idx - 1]

            # Compute end_return (cumulative return after window ends)
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
    """
    Evaluate model using Markovian predictions.

    For each sequence, predicts reward for each (state, action, term) tuple
    independently, then sums predictions to compare with aggregate reward.

    Args:
        model: Trained model
        test_ds: Test dataset
        batch_size: Batch size for evaluation
        regu_lam: Weight applied to the return-consistency regularization MSE when
            computing the total loss.
        collect_predictions: Whether to collect detailed predictions for analysis.
            If False, only MSE is computed (faster, less memory).
        max_batches: Maximum number of batches to evaluate. If None, evaluates
            entire dataset.
        shuffle: Whether to shuffle the test dataloader

    Returns:
        Tuple of (metrics, predictions_list) where metrics is a dict with keys
        "reward", "regu", and "total", each mapping to an array of per-batch MSE
        values. If collect_predictions=False, predictions_list will be empty.
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

            # Predict reward for each (state, action, term) tuple independently
            per_step_predictions = []
            for step_idx in range(seq_len):
                # Extract (state, action, term) at timestep step_idx
                state_t = inputs["state"][:, step_idx, :]  # (batch_size, state_dim)
                action_t = inputs["action"][:, step_idx, :]  # (batch_size, action_dim)
                term_t = inputs["term"][:, step_idx, :]  # (batch_size, 1)

                # Add sequence dimension
                state_seq = state_t.unsqueeze(1)  # (batch_size, 1, state_dim)
                action_seq = action_t.unsqueeze(1)  # (batch_size, 1, action_dim)
                term_seq = term_t.unsqueeze(1)  # (batch_size, 1, 1)

                reward_t = model(state_seq, action_seq, term_seq)
                if reward_t.dim() == 3:
                    reward_t = reward_t.squeeze(1)  # (batch_size, 1)
                per_step_predictions.append(reward_t)

            # Stack predictions: (batch_size, seq_len, 1)
            predictions = torch.stack(per_step_predictions, dim=1)

            # Sum predictions across sequence to get aggregate reward
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
    eval_losses: Sequence,
    final_mse: Mapping[str, float],
    final_rmse: Mapping[str, float],
    reward_model_kwargs: Optional[Mapping[str, Any]] = None,
):
    """
    Save configuration and training metrics to JSON files.

    Args:
        output_dir: Directory to save files
        model_type: Type of model used
        env: Gymnasium environment
        batch_size: Batch size used in training
        eval_steps: Number of evaluation steps
        train_losses: List of training losses per epoch
        eval_losses: List of evaluation losses
        final_mse: Final mean squared error on test set
        final_rmse: Final root mean squared error on test set
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Save training metrics
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

    # Save config
    config_file = os.path.join(output_dir, "config.json")
    config = {
        "spec": SPEC,
        "model_type": model_type,
        "env_name": env.spec.id if hasattr(env, "spec") and env.spec else "unknown",
        "state_dim": obs_dim,
        "action_dim": act_dim,
        "batch_size": batch_size,
        "eval_steps": eval_steps,
        "reward_model_kwargs": reward_model_kwargs or {},
    }
    with tf.io.gfile.GFile(config_file, "w") as writable:
        json.dump(
            config,
            writable,
            indent=2,
        )
    logging.info("Config saved to %s", config_file)

    # TensorBoard add_hparams only accepts scalar types; expand reward_model_kwargs as flat entries
    hparams = {
        key: value for key, value in config.items() if key != "reward_model_kwargs"
    }
    for key, value in (reward_model_kwargs or {}).items():
        hparams[f"reward_model_kwargs.{key}"] = value
    return hparams


def train(
    env: gym.Env,
    dataset: data.Dataset,
    train_epochs: int,
    batch_size: int,
    eval_steps: int,
    log_episode_frequency: int,
    regu_lam: float,
    reward_model_kwargs: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    model_type: str = "mlp",
    output_dir: str = "outputs",
    on_batch_end: Optional[Callable[[Mapping[str, float]], None]] = None,
):
    """
    Train a reward prediction model using Expectation-Maximisation.

    Each batch update performs an E-step (no_grad forward pass to compute per-step
    soft targets) followed by an M-step (backward pass against those targets).

    Args:
        env: Gymnasium environment.
        dataset: Training dataset of (inputs, labels) examples.
        train_epochs: Number of full passes over the training split.
        batch_size: Mini-batch size for training and evaluation.
        eval_steps: Max batches per mid-training evaluation pass.
        log_episode_frequency: Run evaluation every this many epochs.
        regu_lam: Weight for the return-consistency regularization term (λ ∈ [0, 1]).
        seed: Random seed for weight initialisation and data splitting.
        model_type: Model architecture to use ("mlp").
        output_dir: Directory for model checkpoints, metrics, and predictions.
        on_batch_end: Optional callback invoked after every batch update with a dict
            containing {"reward": float, "regu": float, "total": float} loss values.
            Useful for testing and monitoring training dynamics.

    Returns:
        Tuple of (final_mse, predictions_list) where final_mse is a dict with
        keys "reward", "regu", and "total" mapping to scalar MSE values, and
        predictions_list is a list of per-example prediction dicts.
    """
    logging.info("Training with seed: %s", seed)
    torch.manual_seed(seed if seed is not None else 127)
    train_ds, test_ds = data.random_split(dataset, lengths=[0.7, 0.3])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Select model architecture
    if model_type == "mlp":
        model = RNetwork(
            state_dim=obs_dim, action_dim=act_dim, **(reward_model_kwargs or {})
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp'.")

    logging.info("Training with %s model (EM)", model_type.upper())
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    regu_criterion = nn.MSELoss()

    train_losses = []
    eval_losses = []

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
                seq_lengths = labels["seq_lengths"]  # (batch,)
                aggregate_rewards = labels["aggregate_reward"].float()  # (batch,)
                max_seq_len = inputs["state"].shape[1]

                # Mask: True for actual steps, False for padding
                mask = create_sequence_mask(
                    seq_lengths, max_seq_len
                )  # (batch, seq_len)

                # E-step: compute per-step soft targets without gradient
                soft_targets = compute_soft_targets(
                    model, inputs, seq_lengths, aggregate_rewards, mask
                )  # (batch, seq_len)

                # M-step: regress model against soft targets
                outputs = model(**inputs)  # (batch, seq_len, 1)
                per_step_preds = outputs.squeeze(-1)  # (batch, seq_len)

                # Masked per-step MSE against soft targets
                sq_err = (per_step_preds - soft_targets) ** 2  # (batch, seq_len)
                reward_loss = (sq_err * mask.float()).sum() / mask.float().sum()

                # Window reward for return regularization (masked sum over actual steps)
                window_reward = (per_step_preds * mask.float()).sum(dim=1)  # (batch,)
                regu_loss = regu_criterion(
                    labels["start_return"] + window_reward, labels["end_return"]
                )

                loss = reward_loss + (regu_lam * regu_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses["reward"].append(reward_loss.item())
                epoch_losses["regu"].append(regu_loss.item())
                epoch_losses["total"].append(loss.item())
                if on_batch_end is not None:
                    on_batch_end(
                        {
                            "reward": reward_loss.item(),
                            "regu": regu_loss.item(),
                            "total": loss.item(),
                        }
                    )

            # Mean loss for the epoch
            avg_train_loss = np.mean(epoch_losses["total"]).item()
            train_losses.append(avg_train_loss)
            for key in ("reward", "regu", "total"):
                summary_writer.add_scalar(
                    f"MSE/train/{key}", np.mean(epoch_losses[key]), global_step=epoch
                )
                summary_writer.add_scalar(
                    f"RMSE/train/{key}",
                    np.mean(np.sqrt(epoch_losses[key])),
                    global_step=epoch,
                )

            # Evaluation using Markovian predictions
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
                eval_losses.append(np.mean(eval_mse["total"]).item())
                for key in ("reward", "regu", "total"):
                    summary_writer.add_scalar(
                        f"MSE/eval/{key}", np.mean(eval_mse[key]), global_step=epoch
                    )
                    summary_writer.add_scalar(
                        f"RMSE/eval/{key}",
                        np.mean(np.sqrt(eval_mse[key])),
                        global_step=epoch,
                    )
                train_rmse = np.sqrt(avg_train_loss)
                eval_rmse = np.mean(np.sqrt(eval_mse["total"]))
                logging.info(
                    "Epoch [%d/%d], Train RMSE: %.8f, Eval RMSE: %.8f",
                    epoch + 1,
                    train_epochs,
                    train_rmse,
                    eval_rmse,
                )

        # Final evaluation and save predictions
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

        # Save predictions
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

        # Save model
        model_file = os.path.join(output_dir, f"model_{model_type}.pt")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        with tf.io.gfile.GFile(model_file, "wb") as writable:
            writable.write(buffer.read())
        logging.info("Model saved to %s", model_file)

        # Save config and metrics
        hparams = save_config_and_metrics(
            output_dir=output_dir,
            model_type=model_type,
            env=env,
            batch_size=batch_size,
            eval_steps=eval_steps,
            train_losses=train_losses,
            eval_losses=eval_losses,
            final_mse=final_mse,
            final_rmse=final_rmse,
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


def create_sequence_mask(seq_lengths: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Create a boolean mask for valid (non-padded) sequence positions.

    Args:
        seq_lengths: LongTensor of shape (batch,) with the actual length of each sequence.
        max_seq_len: Total (padded) sequence length.

    Returns:
        BoolTensor of shape (batch, max_seq_len) where entry [b, t] is True if
        position t is within the actual sequence for example b.
    """
    position_indices = torch.arange(max_seq_len).unsqueeze(0)  # (1, max_seq_len)
    return position_indices < seq_lengths.unsqueeze(1)  # (batch, max_seq_len)


def compute_soft_targets(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    seq_lengths: torch.Tensor,
    aggregate_rewards: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """E-step: compute per-step soft targets by distributing the window residual.

    For each window w of length D with observed aggregate Yᵥ and current predictions
    μₜ = f_θ(sₜ, aₜ), the soft target is:

        t̃ₜ = μₜ + (Yᵥ - Σⱼ μⱼ) / D

    The correction (Yᵥ - Σⱼ μⱼ) / D is identical for every step in the window,
    following from the equal-variance Gaussian assumption.

    Args:
        model: Current reward model.
        inputs: Dict with tensors 'state', 'action', 'term' of shape
            (batch, max_seq_len, dim).
        seq_lengths: LongTensor of shape (batch,) with actual window lengths.
        aggregate_rewards: FloatTensor of shape (batch,) with observed aggregates.
        mask: BoolTensor of shape (batch, max_seq_len) masking actual positions.

    Returns:
        FloatTensor of shape (batch, max_seq_len). Values at padded positions
        are present but should be excluded via mask in the M-step loss.
    """
    with torch.no_grad():
        outputs_old = model(**inputs)  # (batch, seq_len, 1)
        mu = outputs_old.squeeze(-1)  # (batch, seq_len)
        # Sum only over actual (non-padded) steps
        window_mu = (mu * mask.float()).sum(dim=1)  # (batch,)
        residual = aggregate_rewards - window_mu  # (batch,)
        # Distribute residual uniformly across actual steps
        delta = (residual / seq_lengths.float()).unsqueeze(1)  # (batch, 1)
        return mu + delta.expand_as(mu)  # (batch, seq_len)


def experiment(args: TrainingArgs):
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
            reward_model_kwargs=args.reward_model_kwargs,
            log_episode_frequency=args.log_episode_frequency,
            output_dir=args.output_dir,
            seed=args.seed,
        )


@ray.remote
def run_fn(args: TrainingArgs):
    """
    Wrapper function for `experiment`.
    Allows Ray to report failures.
    """
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


def main():
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


def parse_args() -> TrainingArgs:
    """Parse command-line arguments into a TrainingArgs instance.

    Returns:
        TrainingArgs populated from command-line flags.
    """
    parser = argparse.ArgumentParser(
        description="Train EM reward prediction models for delayed feedback"
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
        help="Number of evaluation steps",
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
        help="Number of runs - with their own seed.",
    )
    parser.add_argument(
        "--regu-lam",
        type=float,
        default=1.0,
        help="Weight of regularisation. lam in [0, 1]",
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


if __name__ == "__main__":
    main()
