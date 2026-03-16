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
import json
import pathlib
import tempfile
from typing import Any, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data, tensorboard

from drmdp import dataproc, rewdelay

# Spec version identifier
SPEC = "o1"


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
        self.layers = nn.ModuleList(layers)
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
        for layer in self.layers:
            out = layer(out)
        return self.final_layer(out)


class DictDataset(data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dictionary for the given index
        return {key: self.inputs[key][idx] for key in self.inputs}, self.labels[idx]


def create_training_buffer(env, delay: rewdelay.RewardDelay, num_steps: int):
    """
    Collects example of (s,a,s',r,d) from an environment.
    """
    buffer = dataproc.collection_traj_data(env, steps=num_steps, include_term=True)
    return delayed_reward_data(buffer, delay=delay)


def delayed_reward_data(buffer, delay: rewdelay.RewardDelay):
    """
    Creates a dataset where each sample corresponds to.
    Assumes actions are continuous or intended to be used as is.
    """

    def create_traj_step(state, action, reward, term):
        return {
            "state": torch.tensor(state),
            "action": torch.tensor(action),
            "term": torch.tensor([float(term)]),
        }, torch.tensor(reward, dtype=torch.float32)

    def create_example(traj_steps: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        # assumes labels single value tensors
        inputs, labels = zip(*traj_steps)

        # Extract per-step rewards and compute aggregate
        per_step_rewards = [label.item() for label in labels]
        aggregate_reward = sum(per_step_rewards)

        # Return dict with both aggregate and per-step rewards
        label_dict = {
            "aggregate_reward": torch.tensor(aggregate_reward),
            "per_step_rewards": torch.tensor(per_step_rewards),
        }

        return data.default_collate(inputs), label_dict

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
    examples = []
    idx = 0
    while True:
        example_steps = []
        steps = 0
        reward_delay = delay.sample()
        while True:
            traj_step = create_traj_step(
                states[idx][:obs_dim], action[idx], reward[idx], term[idx]
            )
            example_steps.append(traj_step)
            idx += 1
            steps += 1
            if steps == reward_delay or (idx >= n_steps):
                break
        # example is complete:
        if steps == reward_delay:
            examples.append(create_example(example_steps))
        if idx >= n_steps:
            break
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
    test_dataloader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)
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
            pred_window_reward = torch.squeeze(torch.sum(predictions, dim=1))

            # Extract aggregate rewards from batched label dict
            # DataLoader collates dicts, so labels["aggregate_reward"] is already a tensor
            aggregate_rewards = labels["aggregate_reward"].float()
            mean_squared_error = eval_criterion(pred_window_reward, aggregate_rewards)
            errors.append(mean_squared_error)

            # Optionally collect predictions for analysis
            if collect_predictions:
                for idx in range(batch_size_actual):
                    per_step_rewards = labels["per_step_rewards"][idx].cpu().numpy()
                    predictions_list.append(
                        {
                            "state": inputs["state"][idx].cpu().numpy(),
                            "action": inputs["action"][idx].cpu().numpy(),
                            "term": inputs["term"][idx].cpu().numpy(),
                            "actual_reward": aggregate_rewards[idx].item(),
                            "per_step_rewards": per_step_rewards,
                            "predicted_reward": pred_window_reward[idx].item(),
                            "per_step_predictions": predictions[idx]
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


def train(
    env: gym.Env,
    dataset: data.Dataset,
    train_epochs: int,
    batch_size: int,
    eval_steps: int,
    log_episode_frequency: int,
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
    train_ds, test_ds = data.random_split(dataset, lengths=[0.7, 0.3])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Select model architecture
    if model_type == "mlp":
        model = RNetwork(state_dim=obs_dim, action_dim=act_dim, hidden_dim=256)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp'.")

    print(f"Training with {model_type.upper()} model")
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # Training Loop
    global_step = 0
    train_losses = []
    eval_losses = []

    with tensorboard.SummaryWriter(log_dir=output_dir) as summary_writer:
        for epoch in range(train_epochs):
            train_dataloader = data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True
            )
            # Training
            epoch_losses = []
            for inputs, labels in train_dataloader:
                # Forward pass (models handle zero stub internally for RNN/Transformer)
                outputs = model(**inputs)

                # Calculate loss for each seq in batch
                # outputs shape: (batch_size, seq_len, 1)
                window_reward = torch.squeeze(torch.sum(outputs, dim=1))

                # Extract aggregate rewards from batched label dict
                # DataLoader collates dicts, so labels["aggregate_reward"] is already a tensor
                aggregate_rewards = labels["aggregate_reward"].float()
                loss = criterion(window_reward, aggregate_rewards)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                global_step += 1

            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)
            # Epoch results
            summary_writer.add_scalar("Loss/train", loss, global_step=epoch)

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
                summary_writer.add_scalar("MSE/eval", eval_mse, global_step=epoch)
                summary_writer.add_scalar("RMSE/eval", eval_rmse, global_step=epoch)
                print(
                    f"Epoch [{epoch + 1}/{train_epochs}], Train RMSE: {train_rmse:.8f}, Eval RMSE: {eval_rmse:.8f}"
                )

        # Final evaluation and save predictions
        print("\nFinal evaluation...")
        final_mse, predictions_list = evaluate_model(model, test_ds, batch_size)
        final_rmse = np.sqrt(final_mse)
        print(f"Final Test RMSE: {final_rmse:.8f}")

        # Save predictions
        predictions_file = pathlib.Path(output_dir) / f"predictions_{model_type}.json"
        with open(predictions_file, "w", encoding="UTF-8") as writable:
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
        print(f"Predictions saved to {predictions_file}")

        # Save training metrics
        metrics_file = pathlib.Path(output_dir) / f"metrics_{model_type}.json"
        with open(metrics_file, "w", encoding="UTF-8") as writable:
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
        print(f"Training metrics saved to {metrics_file}")

        # Save model
        model_file = pathlib.Path(output_dir) / f"model_{model_type}.pt"
        torch.save(model.state_dict(), model_file)
        print(f"Model saved to {model_file}")

        # Save config
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        config_file = pathlib.Path(output_dir) / "config.json"
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
        with open(config_file, "w", encoding="UTF-8") as writable:
            json.dump(
                hparams,
                writable,
                indent=2,
            )
        print(f"Config saved to {config_file}")

        # Save config and final metrics
        summary_writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={"MSE/eval": final_mse, "RMSE/eval": final_rmse},
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


def main():
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

    args, _ = parser.parse_known_args()

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
    inputs, labels = zip(*training_buffer)
    # Labels are dicts - pass them as-is
    dataset = DictDataset(data.default_collate(inputs), list(labels))

    # Train model(s)
    if args.model_type == "mlp":
        print("\n" + "=" * 80)
        print("Training MLP")
        train(
            env,
            dataset=dataset,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            eval_steps=args.eval_steps,
            model_type=args.model_type,
            log_episode_frequency=args.log_episode_frequency,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
