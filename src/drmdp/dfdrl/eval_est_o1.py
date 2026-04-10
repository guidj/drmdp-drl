"""
Evaluation module for O1 (delayed aggregate reward prediction) models.

Loads saved models from est_o1.py and evaluates them in two modes:
1. Predictions mode: Loads and displays predictions from saved JSON
2. Interactive mode: Runs live environment rollouts with delayed rewards
"""

import argparse
import io
import json
import os
import typing
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import tensorflow as tf
import torch
from torch import nn

from drmdp import rewdelay
from drmdp.dfdrl import est_o1


def load_config(output_dir: str) -> Dict[str, Any]:
    """
    Load configuration from saved model directory.

    Args:
        output_dir: Directory containing config.json

    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(output_dir, "config.json")
    if not tf.io.gfile.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with tf.io.gfile.GFile(config_path, "r") as readable:
        config = typing.cast(Dict[str, Any], json.load(readable))

    return config


def load_model(
    model_path: str,
    model_type: str,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
) -> nn.Module:
    """
    Load O1 model from checkpoint.

    Args:
        model_path: Path to model_{model_type}.pt file
        model_type: Type of model ("mlp",)
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension (default: 256)

    Returns:
        Loaded model in eval mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct model architecture based on type
    model: nn.Module
    if model_type == "mlp":
        model = est_o1.RNetwork(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp'.")

    # Load state dict
    if not tf.io.gfile.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load from GFile into BytesIO buffer for torch.load compatibility
    with tf.io.gfile.GFile(model_path, "rb") as readable:
        buffer = io.BytesIO(readable.read())
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def evaluate_from_predictions_file(
    predictions_path: str, num_examples: int = 10
) -> None:
    """
    Load and display predictions from saved JSON file.

    Shows aggregate reward predictions for windows (not individual steps).

    Args:
        predictions_path: Path to predictions_{model_type}.json
        num_examples: Number of examples to display (default: 10)
    """
    if not tf.io.gfile.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with tf.io.gfile.GFile(predictions_path, "r") as readable:
        data = json.load(readable)

    predictions = data["predictions"]
    num_to_show = min(num_examples, len(predictions))

    print("\n" + "=" * 80)
    print(
        f"O1 Model Predictions ({data['model_type'].upper()}) - showing {num_to_show}/{len(predictions)} examples"
    )
    print("=" * 80)
    print(
        f"{'Window':>8s} | {'Seq Len':>8s} | {'Actual Agg':>20s} | {'Predicted Agg':>20s} | {'Error':>20s}"
    )
    print("=" * 80)

    errors = []
    for idx in range(num_to_show):
        pred = predictions[idx]
        seq_len = len(pred["per_step_predictions"])
        actual = pred["actual_reward"]
        predicted = pred["predicted_reward"]
        error = abs(actual - predicted)
        errors.append(error)

        print(
            f"{idx:8d} | {seq_len:8d} | {actual:20.8f} | {predicted:20.8f} | {error:20.8f}"
        )

    print("=" * 80)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    print(f"Mean Absolute Error: {mae:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"Overall MSE (from file): {data['final_mse']:.8f}")
    print("=" * 80)


def evaluate_interactive(
    model: nn.Module,
    env: gym.Env,
    delay: rewdelay.RewardDelay,
    num_episodes: int = 5,
) -> None:
    """
    Run live environment rollouts with delayed rewards and compare predictions.

    Collects sequences of (s,a) pairs, sums predicted rewards, and compares
    to aggregate feedback after delay.

    Args:
        model: Loaded O1 model
        env: Gymnasium environment
        delay: Reward delay distribution
        num_episodes: Number of episodes to run (default: 5)
    """
    device = next(model.parameters()).device
    all_errors = []

    print("\n" + "=" * 80)
    print(f"Interactive Evaluation - Running {num_episodes} episodes with delay")
    print("=" * 80)

    total_windows = 0

    for episode_idx in range(num_episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        window_count = 0

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")
        print("-" * 80)
        print(
            f"{'Window':>8s} | {'Seq Len':>8s} | {'Actual Agg':>20s} | {'Predicted Agg':>20s} | {'Error':>20s}"
        )
        print("-" * 80)

        while not (terminated or truncated):
            # Collect sequence based on delay
            sequence_states = []
            sequence_actions = []
            sequence_terms = []
            actual_rewards = []

            window_delay = delay.sample()
            step_count = 0

            while step_count < window_delay and not (terminated or truncated):
                action = env.action_space.sample()

                sequence_states.append(observation)
                sequence_actions.append(action)
                sequence_terms.append(float(terminated))

                next_observation, reward, terminated, truncated, _ = env.step(action)
                actual_rewards.append(reward)

                observation = next_observation
                step_count += 1

            # Only evaluate if we have a full window
            if step_count == window_delay:
                # Prepare batch
                states_tensor = (
                    torch.tensor(sequence_states, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .to(device)
                )
                actions_tensor = (
                    torch.tensor(sequence_actions, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .to(device)
                )
                terms_tensor = (
                    torch.tensor(
                        [[t] for t in sequence_terms],
                        dtype=torch.float32,
                        device=device,
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                # Predict rewards and sum
                with torch.no_grad():
                    per_step_preds = model(
                        states_tensor, actions_tensor, terms_tensor
                    )  # (1, seq_len, 1)
                    predicted_aggregate = per_step_preds.sum().item()

                # Compute actual aggregate
                actual_aggregate = sum(actual_rewards)

                # Compute error
                error = abs(actual_aggregate - predicted_aggregate)
                all_errors.append(error)

                # Display first 20 windows
                if window_count < 20:
                    print(
                        f"{window_count:8d} | {step_count:8d} | {actual_aggregate:20.8f} | {predicted_aggregate:20.8f} | {error:20.8f}"
                    )
                elif window_count == 20:
                    print("..." + " " * 77)

                window_count += 1
                total_windows += 1

        print("-" * 80)
        if window_count > 0:
            episode_mae = np.mean(
                all_errors[-window_count:]
            )  # Last window_count errors
            print(
                f"Episode {episode_idx + 1} MAE: {episode_mae:.8f} ({window_count} windows)"
            )
        else:
            print(f"Episode {episode_idx + 1}: No complete windows")

    print("=" * 80)
    if all_errors:
        overall_mae = np.mean(all_errors)
        overall_rmse = np.sqrt(np.mean(np.square(all_errors)))
        print(f"Overall MAE: {overall_mae:.8f} ({total_windows} windows)")
        print(f"Overall RMSE: {overall_rmse:.8f}")
    else:
        print("No complete windows collected")
    print("=" * 80)


def main():
    """Parse args and evaluate O1 delayed aggregate reward prediction models."""
    parser = argparse.ArgumentParser(
        description="Evaluate O1 (delayed aggregate reward prediction) models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing saved model (e.g., outputs/o1/1709564425)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["mlp"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="predictions",
        choices=["predictions", "interactive"],
        help="Evaluation mode: predictions (from file) or interactive (live env)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to display in predictions mode (default: 10)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment for interactive mode (defaults to config env)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Fixed delay for interactive mode (default: 3)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes for interactive mode (default: 5)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2500,
        help="Max steps per episode (default: 2500)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.model_dir)
    print(f"Loaded config from {args.model_dir}")
    print(f"Spec: {config['spec']}")
    print(f"Model type: {config['model_type']}")
    print(f"Environment: {config['env_name']}")

    # Verify model type matches
    if args.model_type != config["model_type"]:
        print(
            f"Warning: Requested model_type '{args.model_type}' differs from config '{config['model_type']}'"
        )
        print(f"Using requested model_type: {args.model_type}")

    if args.mode == "predictions":
        # Evaluate from predictions file
        predictions_path = os.path.join(
            args.model_dir, f"predictions_{args.model_type}.json"
        )
        evaluate_from_predictions_file(predictions_path, args.num_examples)

    elif args.mode == "interactive":
        # Load model
        model_path = os.path.join(args.model_dir, f"model_{args.model_type}.pt")
        model = load_model(
            model_path,
            model_type=args.model_type,
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config.get("hidden_dim", 256),
        )
        print(f"Loaded {args.model_type.upper()} model from {model_path}")

        # Create environment
        env_name = args.env if args.env else config["env_name"]
        env = gym.make(env_name, max_episode_steps=args.max_episode_steps)
        print(f"Created environment: {env_name}")

        # Create delay
        delay = rewdelay.FixedDelay(args.delay)
        print(f"Using fixed delay: {args.delay}")

        # Run interactive evaluation
        evaluate_interactive(model, env, delay, args.num_episodes)

        env.close()


if __name__ == "__main__":
    main()
