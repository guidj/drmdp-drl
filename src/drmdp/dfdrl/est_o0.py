"""
Reward prediction for individual (state, action) pairs.

This module provides a simpler baseline for reward prediction where:
- Input: A single (state, action) pair
- Output: The immediate reward for that state-action pair

This is the Markovian baseline (o=0) compared to the delayed aggregate feedback
models in est_o1.py (o=1).

Usage Example:
    # Train model on immediate reward prediction
    python est_o0.py --env MountainCarContinuous-v0 --num-steps 100000

    # Customize training
    python est_o0.py --env Pendulum-v1 --num-steps 50000 --batch-size 128
"""

import argparse
import json
import pathlib
import time
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from drmdp import dataproc

# Spec version identifier
SPEC = "o0"


def create_timestamped_output_dir(base_dir: str) -> pathlib.Path:
    """Create versioned timestamped output directory: {base_dir}/{SPEC}/{unix_timestamp}/"""
    timestamp = int(time.time())
    output_path = pathlib.Path(base_dir) / SPEC / str(timestamp)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


class RNetwork(nn.Module):
    """
    Feedforward MLP for immediate reward prediction.
    Maps (state, action, term) -> reward.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # +1 for term flag
        self.fc1 = nn.Linear(state_dim + action_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, term):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim)
            term: Tensor of shape (batch_size, 1)

        Returns:
            reward: Tensor of shape (batch_size, 1)
        """
        features = torch.concat([state, action, term], dim=-1)
        features = nn.functional.relu(self.fc1(features))
        features = nn.functional.relu(self.fc2(features))
        reward = self.fc3(features)
        return reward


class DictDataset(data.Dataset):
    """Dataset wrapper for dictionary-based inputs."""

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return a dictionary of inputs and the corresponding label."""
        return {key: self.inputs[key][idx] for key in self.inputs}, self.labels[idx]


def create_training_buffer(env, num_steps: int):
    """
    Collects individual (s, a, d, r) tuples from an environment.

    Args:
        env: Gymnasium environment
        num_steps: Number of steps to collect

    Returns:
        List of tuples (inputs_dict, reward) where inputs_dict contains
        "state", "action", and "term" tensors
    """
    buffer = dataproc.collection_traj_data(env, steps=num_steps, include_term=True)
    return immediate_reward_data(buffer)


def immediate_reward_data(buffer):
    """
    Creates a dataset of individual (state, action, term) -> reward mappings.

    Args:
        buffer: List of (state, action, next_state, reward, term) tuples

    Returns:
        List of (inputs_dict, reward) tuples
    """
    examples = []

    for example in buffer:
        state = example[0]
        action = example[1]
        reward = example[3]
        term = example[4]

        inputs = {
            "state": torch.tensor(state, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "term": torch.tensor([float(term)], dtype=torch.float32),
        }
        label = torch.tensor(reward, dtype=torch.float32)
        examples.append((inputs, label))

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
    Evaluate model on individual (state, action) pairs.

    Args:
        model: Trained model
        test_ds: Test dataset
        batch_size: Batch size for evaluation
        collect_predictions: Whether to collect detailed predictions for analysis.
            If False, only MSE is computed (faster, less memory).
            When True, ensures at least one example with term=True is collected.
        max_batches: Maximum number of batches to evaluate. If None, evaluates
            entire dataset. Useful for quick checks during training.
            Ignored when collect_predictions=True to ensure term=True examples.
        shuffle: Whether to shuffle the test dataloader

    Returns:
        Tuple of (mean_squared_error, predictions_list)
        If collect_predictions=False, predictions_list will be empty.
    """
    test_dataloader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)
    eval_criterion = nn.MSELoss()
    errors = []
    predictions_list = []

    # When collecting predictions, evaluate entire dataset to ensure term=True examples
    effective_max_batches = None if collect_predictions else max_batches

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            # Forward pass: predict reward for each (state, action) pair
            predictions = model(**inputs).squeeze(-1)  # (batch_size,)

            # Compute MSE
            mean_squared_error = eval_criterion(predictions, labels)
            errors.append(mean_squared_error)

            # Optionally collect predictions for analysis
            if collect_predictions:
                batch_size_actual = inputs["state"].shape[0]
                for idx in range(batch_size_actual):
                    predictions_list.append(
                        {
                            "state": inputs["state"][idx].cpu().numpy(),
                            "action": inputs["action"][idx].cpu().numpy(),
                            "term": inputs["term"][idx].cpu().numpy(),
                            "actual_reward": labels[idx].item(),
                            "predicted_reward": predictions[idx].item(),
                        }
                    )

            # Stop early if max_batches is specified (but not when collecting predictions)
            if effective_max_batches is not None and idx + 1 >= effective_max_batches:
                break

    # When collecting predictions, verify we have at least one term=True example
    if collect_predictions and predictions_list:
        has_term_true = any(pred["term"][0] > 0.5 for pred in predictions_list)
        term_count = sum(1 for pred in predictions_list if pred["term"][0] > 0.5)
        print(
            f"Collected {len(predictions_list)} predictions ({term_count} with term=True)"
        )
        if not has_term_true:
            print("Warning: No examples with term=True found in evaluation set")

    mse = torch.mean(torch.stack(errors)).item()
    return mse, predictions_list


def train(
    env: gym.Env,
    dataset: data.Dataset,
    batch_size: int,
    eval_steps: int,
    output_dir: str = "outputs",
):
    """
    Train an immediate reward prediction model.

    Args:
        env: Gymnasium environment
        dataset: Training dataset
        batch_size: Batch size for training
        eval_steps: Number of evaluation steps
        output_dir: Directory to save predictions and results

    Returns:
        Tuple of (final_mse, predictions_list)
    """
    train_ds, test_ds = data.random_split(dataset, lengths=[0.7, 0.3])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create model
    model = RNetwork(state_dim=obs_dim, action_dim=act_dim, hidden_dim=256)
    print("Training immediate reward prediction model (MLP)")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training Loop
    epochs = 100
    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        train_dataloader = data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )

        # Training
        epoch_losses = []
        for inputs, labels in train_dataloader:
            # Forward pass
            outputs = model(**inputs).squeeze(-1)  # (batch_size,)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        # Evaluation
        if (epoch + 1) % 5 == 0:
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
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train RMSE: {train_rmse:.8f}, Eval RMSE: {eval_rmse:.8f}"
            )

    # Final evaluation and save predictions
    print("\nFinal evaluation...")
    final_mse, predictions_list = evaluate_model(model, test_ds, batch_size)
    final_rmse = np.sqrt(final_mse)
    print(f"Final Test RMSE: {final_rmse:.8f}")

    # Save results
    output_path = create_timestamped_output_dir(output_dir)

    # Save predictions
    predictions_file = output_path / "predictions_o0.json"
    with open(predictions_file, "w", encoding="UTF-8") as writable:
        json.dump(
            {
                "model_type": "mlp_immediate",
                "final_mse": final_mse,
                "num_predictions": len(predictions_list),
                "predictions": [
                    {
                        "state": pred["state"].tolist(),
                        "action": pred["action"].tolist(),
                        "term": pred["term"].tolist(),
                        "actual_reward": pred["actual_reward"],
                        "predicted_reward": pred["predicted_reward"],
                    }
                    for pred in predictions_list
                ],
            },
            writable,
            indent=2,
        )
    print(f"Predictions saved to {predictions_file}")

    # Save training metrics
    metrics_file = output_path / "metrics_o0.json"
    with open(metrics_file, "w", encoding="UTF-8") as writable:
        json.dump(
            {
                "model_type": "mlp_immediate",
                "train_losses": train_losses,
                "eval_losses": eval_losses,
                "final_mse": final_mse,
            },
            writable,
            indent=2,
        )
    print(f"Training metrics saved to {metrics_file}")

    # Save model
    model_file = output_path / "model_o0.pt"
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

    # Save config
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    config_file = output_path / "config.json"
    with open(config_file, "w", encoding="UTF-8") as writable:
        json.dump(
            {
                "spec": SPEC,
                "model_type": "mlp",
                "env_name": env.spec.id
                if hasattr(env, "spec") and env.spec
                else "unknown",
                "state_dim": obs_dim,
                "action_dim": act_dim,
                "batch_size": batch_size,
                "eval_steps": eval_steps,
                "hidden_dim": 256,
                "timestamp": int(output_path.name),
            },
            writable,
            indent=2,
        )
    print(f"Config saved to {config_file}")

    return final_mse, predictions_list


def main():
    parser = argparse.ArgumentParser(
        description="Train immediate reward prediction model (Markovian baseline)"
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

    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env, max_episode_steps=args.max_episode_steps)
    print(f"Environment: {args.env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create training buffer
    print(f"\nCollecting {args.num_steps} steps...")
    training_buffer = create_training_buffer(env, num_steps=args.num_steps)
    print(f"Created {len(training_buffer)} training examples")

    # Create dataset
    inputs, labels = zip(*training_buffer)
    dataset = DictDataset(data.default_collate(inputs), torch.stack(labels))

    # Train model
    train(
        env,
        dataset=dataset,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
