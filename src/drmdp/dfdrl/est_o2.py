"""
Reward prediction networks for delayed, aggregate, and anonymous feedback (DAAF).

This module provides three architectures for predicting rewards from sequences
of (state, action) pairs:

1. RNetwork: Feedforward MLP - processes each (s,a) pair independently

All models support two modes:
- forward(): Sequential prediction for training on delayed reward sequences
- forward_markovian(): Markovian prediction for test time (reward depends only on current s,a)

Usage Example:
    # Feedforward (Each step is independent)
    model = RNetwork(state_dim=4, action_dim=2, hidden_dim=256)
    # The last variable indicates termination
    # Sequential input: (batch_size, seq_len, state_dim), (batch_size, seq_len, action_dim), (batch_size, seq_len, 1)
    rewards = model(states, actions)  # Output: (batch_size, seq_len, 1)
    # Markovian input: (batch_size, state_dim), (batch_size, action_dim), (batch_size, 1)
    reward = model.forward_markovian(state, action)  # Output: (batch_size, 1)

All models output per-step reward predictions that are summed during training
to match the delayed aggregate reward signal.
"""

import argparse
import dataclasses
import io
import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import ray
import tensorflow as tf
import torch
from torch import nn, optim
from torch.utils import data, tensorboard

from drmdp import dataproc, ray_utils, rewdelay

# Spec version identifier
SPEC = "o2"


@dataclasses.dataclass(frozen=True)
class TrainingArgs:
    """Arguments for training reward prediction models."""

    model_type: str
    env: str
    max_episode_steps: int
    delay: int
    train_epochs: int
    num_steps: int
    batch_size: int
    eval_steps: int
    log_episode_frequency: int
    output_dir: str
    num_runs: int
    seed: Optional[int] = None


class RNetwork(nn.Module):
    """
    Feedforward MLP for reward prediction.
    Processes (state, action, term) tuples independently.
    """

    def __init__(
        self, state_dim, action_dim, powers=1, num_hidden_layers=4, hidden_dim=256
    ):
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
            reward: Tensor of shape (batch_size, 1)
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
        return self.length

    def __getitem__(self, idx):
        # Return raw example without collation
        return self.inputs[idx], self.labels[idx]


def collate_variable_length_sequences(batch):
    """
    Custom collate function for batching variable-length sequences.
    Pads sequences to the maximum length within the batch.

    Args:
        batch: List of (inputs_dict, labels_dict) tuples

    Returns:
        (batched_inputs_dict, batched_labels_dict) with padded sequences
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
    env, delay: rewdelay.RewardDelay, num_steps: int, seed: Optional[int] = None
):
    """
    Collects example of (s,a,s',r,d) from an environment.
    """
    buffer = dataproc.collection_traj_data(
        env, steps=num_steps, include_term=True, seed=seed
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
            - labels: Dict with 'aggregate_reward' (sum) and 'per_step_rewards' (list)

    Note:
        Assumes actions are continuous or intended to be used as is.
        Sequences shorter than the sampled delay are discarded.
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
    collect_predictions: bool = True,
    max_batches: Optional[int] = None,
    shuffle: bool = False,
) -> Tuple[float, List[Any]]:
    """
    Evaluate model using Markovian predictions.

    For each sequence, predicts reward for each (state, action, term) tuple independently,
    then sums predictions to compare with aggregate reward.

    Args:
        model: Trained model
        test_ds: Test dataset
        batch_size: Batch size for evaluation
        collect_predictions: Whether to collect detailed predictions for analysis.
            If False, only MSE is computed (faster, less memory).
        max_batches: Maximum number of batches to evaluate. If None, evaluates
            entire dataset. Useful for quick checks during training.
        shuffle: Whether to shuffle the test dataloader

    Returns:
        Tuple of (mean_squared_error, predictions_list)
        If collect_predictions=False, predictions_list will be empty.
    """
    test_dataloader = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_length_sequences,
    )
    eval_criterion = nn.MSELoss()
    errors = []
    predictions_list = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            batch_size_actual = inputs["state"].shape[0]
            seq_len = inputs["state"].shape[1]

            # Predict reward for each (state, action, term) tuple independently
            per_step_predictions = []
            for step_idx in range(seq_len):
                # Extract (state, action, term) at timestep step_idx for all sequences in batch
                state_t = inputs["state"][:, step_idx, :]  # (batch_size, state_dim)
                action_t = inputs["action"][:, step_idx, :]  # (batch_size, action_dim)
                term_t = inputs["term"][:, step_idx, :]  # (batch_size, 1)

                # Add sequence dimension for RNN/Transformer models
                state_seq = state_t.unsqueeze(1)  # (batch_size, 1, state_dim)
                action_seq = action_t.unsqueeze(1)  # (batch_size, 1, action_dim)
                term_seq = term_t.unsqueeze(1)  # (batch_size, 1, 1)

                # Forward pass - returns (batch_size, 1, 1) for RNN/Transformer, (batch_size, 1) for MLP
                reward_t = model(state_seq, action_seq, term_seq)
                if reward_t.dim() == 3:
                    reward_t = reward_t.squeeze(1)  # (batch_size, 1)
                per_step_predictions.append(reward_t)

            # Stack predictions: (batch_size, seq_len, 1)
            predictions = torch.stack(per_step_predictions, dim=1)

            # Sum predictions across sequence to get aggregate reward
            # Use squeeze(-1) to only squeeze the last dimension, preserving batch dimension
            pred_window_reward = torch.sum(predictions, dim=1).squeeze(-1)

            # Extract aggregate rewards from batched label dict
            # DataLoader collates dicts, so labels["aggregate_reward"] is already a tensor
            aggregate_rewards = labels["aggregate_reward"].float()
            mean_squared_error = eval_criterion(pred_window_reward, aggregate_rewards)
            errors.append(mean_squared_error)

            # Optionally collect predictions for analysis
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

            # Stop early if max_batches is specified
            if max_batches is not None and idx + 1 >= max_batches:
                break

    mse = torch.mean(torch.stack(errors)).item()
    return mse, predictions_list


def save_config_and_metrics(
    output_dir: str,
    model_type: str,
    env: gym.Env,
    batch_size: int,
    eval_steps: int,
    train_losses: list,
    eval_losses: list,
    final_mse: float,
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
            },
            writable,
            indent=2,
        )
    logging.info("Training metrics saved to %s", metrics_file)

    # Save config
    config_file = os.path.join(output_dir, "config.json")
    hparams = {
        "spec": SPEC,
        "model_type": model_type,
        "env_name": env.spec.id if hasattr(env, "spec") and env.spec else "unknown",
        "state_dim": obs_dim,
        "action_dim": act_dim,
        "batch_size": batch_size,
        "eval_steps": eval_steps,
        "hidden_dim": 256,
    }
    with tf.io.gfile.GFile(config_file, "w") as writable:
        json.dump(
            hparams,
            writable,
            indent=2,
        )
    logging.info("Config saved to %s", config_file)
    return hparams


def train(
    env: gym.Env,
    dataset: data.Dataset,
    train_epochs: int,
    batch_size: int,
    eval_steps: int,
    log_episode_frequency: int,
    seed: Optional[int] = None,
    model_type: str = "mlp",
    output_dir: str = "outputs",
):
    """
    Train a reward prediction model.

    Args:
        env: Gymnasium environment
        dataset: Training dataset
        batch_size: Batch size for training
        eval_steps: Number of evaluation steps
        model_type: Type of model to use ("mlp", "rnn", or "transformer")
        output_dir: Directory to save predictions and results
    """
    logging.info("Training with seed: %s", seed)
    torch.manual_seed(seed if seed is not None else 127)
    train_ds, test_ds = data.random_split(dataset, lengths=[0.7, 0.3])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Select model architecture
    if model_type == "mlp":
        model = RNetwork(state_dim=obs_dim, action_dim=act_dim, hidden_dim=256)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp'.")

    logging.info("Training with %s model", model_type.upper())
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # Training Loop
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
            # Training
            epoch_losses = []
            for inputs, labels in train_dataloader:
                # Forward pass (models handle zero stub internally for RNN/Transformer)
                outputs = model(**inputs)

                # Calculate loss for each seq in batch
                # outputs shape: (batch_size, seq_len, 1)
                # Use squeeze(-1) to only squeeze the last dimension, preserving batch dimension
                window_reward = torch.sum(outputs, dim=1).squeeze(-1)

                # Extract aggregate rewards from batched label dict
                # DataLoader collates dicts, so labels["aggregate_reward"] is already a tensor
                aggregate_rewards = labels["aggregate_reward"].float()
                loss = criterion(window_reward, aggregate_rewards)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)
            # Epoch results
            summary_writer.add_scalar(
                "Loss/train", np.mean(epoch_losses), global_step=epoch
            )

            # Evaluation using Markovian predictions
            if (epoch + 1) % log_episode_frequency == 0:
                eval_mse, _ = evaluate_model(
                    model,
                    test_ds,
                    batch_size=batch_size,
                    collect_predictions=False,
                    max_batches=eval_steps,
                    shuffle=True,
                )
                eval_losses.append(eval_mse)
                train_rmse = np.sqrt(avg_train_loss)
                eval_rmse = np.sqrt(eval_mse)
                # Epoch results
                summary_writer.add_scalar(
                    "MSE/eval", np.mean(eval_losses), global_step=epoch
                )
                summary_writer.add_scalar(
                    "RMSE/eval", np.mean(np.sqrt(eval_losses)), global_step=epoch
                )
                logging.info(
                    "Epoch [%d/%d], Train RMSE: %.8f, Eval RMSE: %.8f",
                    epoch + 1,
                    train_epochs,
                    train_rmse,
                    eval_rmse,
                )

        # Final evaluation and save predictions
        logging.info("Final evaluation...")
        final_mse, predictions_list = evaluate_model(model, test_ds, batch_size)
        final_rmse = np.sqrt(final_mse)
        logging.info("Final Test RMSE: %.8f", final_rmse)

        # Save predictions
        predictions_file = os.path.join(output_dir, f"predictions_{model_type}.json")
        with tf.io.gfile.GFile(predictions_file, "w") as writable:
            json.dump(
                {
                    "model_type": model_type,
                    "final_mse": final_mse,
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
        # torch.save requires a real file path, not a file handle
        # First save to temporary buffer, then write to GFile
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
        )

        # Save config and final metrics
        summary_writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={"MSE": final_mse, "RMSE": final_rmse},
        )
        # Create sample input for graph tracing
        sample_input = (
            # state
            torch.zeros(1, 1, obs_dim),
            # action
            torch.zeros(1, 1, act_dim),
            # term
            torch.zeros(1, 1, 1),
        )
        summary_writer.add_graph(model, sample_input)

    return final_mse, predictions_list


def experiment(args: TrainingArgs):
    logging.basicConfig(level=logging.INFO)
    # Create environment
    env = gym.make(args.env, max_episode_steps=args.max_episode_steps)
    logging.info("Spec: %s", args)

    # Create delay and training buffer
    delay = rewdelay.ClippedPoissonDelay(args.delay, min_delay=2)
    _, max_delay = delay.range()
    logging.info(
        "Collecting %d steps with delay=%d...", args.num_steps * max_delay, args.delay
    )
    training_buffer = create_training_buffer(
        env, delay=delay, num_steps=args.num_steps * max_delay, seed=args.seed
    )
    logging.info("Created %d training examples", len(training_buffer))

    # Create dataset
    inputs, labels = zip(*training_buffer)
    # Store raw examples instead of pre-collated tensors
    dataset = DictDataset(inputs=list(inputs), labels=list(labels))

    # Train model(s)
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
        # Fail job for ray status reporting
        sys.exit(1)


def main():
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
            tasks.append(run_fn.remote(seed_args))

        ray_utils.wait_till_completion(tasks)


def parse_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(
        description="Train reward prediction models for delayed feedback"
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
        "--num-steps",
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

    args, _ = parser.parse_known_args()
    return TrainingArgs(**vars(args))


if __name__ == "__main__":
    main()
