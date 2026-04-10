"""
Evaluation module for O0 (immediate reward prediction) models.

Loads saved models from est_o0.py and evaluates them in two modes:
1. Predictions mode: Loads and displays predictions from saved JSON
2. Interactive mode: Runs live environment rollouts

Usage Examples:
    # Evaluate from saved predictions
    python -m drmdp.dfdrl.eval_est_o0 \
        --model-dir outputs/o0/1709564425 \
        --mode predictions \
        --num-examples 10

    # Interactive evaluation with live environment
    python -m drmdp.dfdrl.eval_est_o0 \
        --model-dir outputs/o0/1709564425 \
        --mode interactive \
        --env MountainCarContinuous-v0 \
        --num-episodes 5
"""

import argparse
import json
import pathlib
import typing
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from drmdp.dfdrl import est_o0


def load_config(output_dir: str) -> Dict[str, Any]:
    """
    Load configuration from saved model directory.

    Args:
        output_dir: Directory containing config.json

    Returns:
        Configuration dictionary
    """
    config_path = pathlib.Path(output_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="UTF-8") as readable:
        config = typing.cast(Dict[str, Any], json.load(readable))

    return config


def load_model(
    model_path: str, state_dim: int, action_dim: int, hidden_dim: int = 256
) -> nn.Module:
    """
    Load O0 model from checkpoint.

    Args:
        model_path: Path to model_o0.pt file
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension (default: 256)

    Returns:
        Loaded model in eval mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct model architecture
    model = est_o0.RNetwork(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
    )

    # Load state dict
    model_path_obj = pathlib.Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path_obj, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def evaluate_from_predictions_file(
    predictions_path: str, num_examples: int = 10
) -> None:
    """
    Load and display predictions from saved JSON file.

    Args:
        predictions_path: Path to predictions_o0.json
        num_examples: Number of examples to display (default: 10)
    """
    predictions_path_obj = pathlib.Path(predictions_path)
    if not predictions_path_obj.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(predictions_path_obj, "r", encoding="UTF-8") as readable:
        data = json.load(readable)

    predictions = data["predictions"]
    num_to_show = min(num_examples, len(predictions))

    print("\n" + "=" * 80)
    print(f"O0 Model Predictions (showing {num_to_show}/{len(predictions)} examples)")
    print("=" * 80)
    print(f"{'Actual Reward':>20s} | {'Predicted Reward':>20s} | {'Error':>20s}")
    print("=" * 80)

    errors = []
    for idx in range(num_to_show):
        pred = predictions[idx]
        actual = pred["actual_reward"]
        predicted = pred["predicted_reward"]
        error = abs(actual - predicted)
        errors.append(error)

        print(f"{actual:20.8f} | {predicted:20.8f} | {error:20.8f}")

    print("=" * 80)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    print(f"Mean Absolute Error: {mae:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"Overall MSE (from file): {data['final_mse']:.8f}")
    print("=" * 80)


def evaluate_interactive(model: nn.Module, env: gym.Env, num_episodes: int = 5) -> None:
    """
    Run live environment rollouts and compare predictions with actual rewards.

    Args:
        model: Loaded O0 model
        env: Gymnasium environment
        num_episodes: Number of episodes to run (default: 5)
    """
    device = next(model.parameters()).device
    all_errors = []

    print("\n" + "=" * 80)
    print(f"Interactive Evaluation - Running {num_episodes} episodes")
    print("=" * 80)

    for episode_idx in range(num_episodes):
        observation, _ = env.reset()
        episode_errors = []
        step_count = 0
        terminated = False
        truncated = False

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")
        print("-" * 80)
        print(
            f"{'Step':>6s} | {'Actual Reward':>20s} | {'Predicted Reward':>20s} | {'Error':>20s}"
        )
        print("-" * 80)

        while not (terminated or truncated):
            # Sample random action
            action = env.action_space.sample()

            # Predict reward
            with torch.no_grad():
                state_tensor = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action_tensor = torch.tensor(
                    action, dtype=torch.float32, device=device
                ).unsqueeze(0)
                term_tensor = torch.tensor(
                    [float(terminated)], dtype=torch.float32, device=device
                ).unsqueeze(0)

                predicted_reward = (
                    model(state_tensor, action_tensor, term_tensor).squeeze().item()
                )

            # Take step in environment
            next_observation, actual_reward, terminated, truncated, _ = env.step(action)

            # Compute error
            error = abs(actual_reward - predicted_reward)
            episode_errors.append(error)
            all_errors.append(error)

            # Display first 20 steps
            if step_count < 20:
                print(
                    f"{step_count:6d} | {actual_reward:20.8f} | {predicted_reward:20.8f} | {error:20.8f}"
                )
            elif step_count == 20:
                print("..." + " " * 77)

            observation = next_observation
            step_count += 1

        episode_mae = np.mean(episode_errors)
        print("-" * 80)
        print(f"Episode {episode_idx + 1} MAE: {episode_mae:.8f} ({step_count} steps)")

    print("=" * 80)
    overall_mae = np.mean(all_errors)
    overall_rmse = np.sqrt(np.mean(np.square(all_errors)))
    print(f"Overall MAE: {overall_mae:.8f}")
    print(f"Overall RMSE: {overall_rmse:.8f}")
    print("=" * 80)


def main():
    """Parse args and evaluate O0 immediate reward prediction models."""
    parser = argparse.ArgumentParser(
        description="Evaluate O0 (immediate reward prediction) models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing saved model (e.g., outputs/o0/1709564425)",
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

    if args.mode == "predictions":
        # Evaluate from predictions file
        predictions_path = pathlib.Path(args.model_dir) / "predictions_o0.json"
        evaluate_from_predictions_file(str(predictions_path), args.num_examples)

    elif args.mode == "interactive":
        # Load model
        model_path = pathlib.Path(args.model_dir) / "model_o0.pt"
        model = load_model(
            str(model_path),
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config.get("hidden_dim", 256),
        )
        print(f"Loaded model from {model_path}")

        # Create environment
        env_name = args.env if args.env else config["env_name"]
        env = gym.make(env_name, max_episode_steps=args.max_episode_steps)
        print(f"Created environment: {env_name}")

        # Run interactive evaluation
        evaluate_interactive(model, env, args.num_episodes)

        env.close()


if __name__ == "__main__":
    main()
