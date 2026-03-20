"""
Evaluation module for O2 reward estimation models.

Loads saved MLP models from est_o2.py and evaluates them.

Evaluation modes:
1. Predictions mode: Loads and displays predictions from saved JSON
2. Interactive mode: Runs live environment rollouts with delayed rewards

Usage Examples:
    # Evaluate from saved predictions
    python -m drmdp.dfdrl.eval_est_o2 \
        --model-dir outputs/o2/mlp \
        --mode predictions \
        --num-examples 10

    # Interactive evaluation with live environment
    python -m drmdp.dfdrl.eval_est_o2 \
        --model-dir outputs/o2/mlp \
        --mode interactive \
        --env MountainCarContinuous-v0 \
        --delay 3 \
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

from drmdp import rewdelay
from drmdp.dfdrl import est_o2


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
    model_path: str,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
) -> nn.Module:
    """
    Load reward model from checkpoint.

    Args:
        model_path: Path to model file (model_mlp.pt)
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension (default: 256)

    Returns:
        Loaded reward model (MLP only)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create reward model (MLP only)
    r_model = est_o2.RNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    # Load checkpoint
    model_path_obj = pathlib.Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path_obj, map_location=device, weights_only=False)
    r_model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from: {model_path_obj}")

    r_model.to(device)
    r_model.eval()

    return r_model


def evaluate_from_predictions_file(
    predictions_path: str, num_examples: int = 10
) -> None:
    """
    Load and display predictions from saved JSON file.

    Args:
        predictions_path: Path to predictions_mlp.json
        num_examples: Number of examples to display (default: 10)
    """
    predictions_path_obj = pathlib.Path(predictions_path)
    if not predictions_path_obj.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(predictions_path_obj, "r", encoding="UTF-8") as readable:
        data = json.load(readable)

    predictions = data["predictions"]
    num_to_show = min(num_examples, len(predictions))

    print("\n" + "=" * 100)
    print(f"O2 Model Predictions (Reward: {data['model_type'].upper()})")
    print(f"Showing {num_to_show}/{len(predictions)} examples")
    print("=" * 100)
    print(
        f"{'Window':>8s} | "
        f"{'Length':>8s} | "
        f"{'Actual Reward':>15s} | "
        f"{'Predicted Reward':>15s} | "
        f"{'Error':>15s}"
    )
    print("=" * 100)

    reward_errors = []

    for idx in range(num_to_show):
        pred = predictions[idx]

        window_len = len(pred["state"])
        actual_reward = pred["actual_reward"]
        predicted_reward = pred["predicted_reward"]
        reward_error = abs(actual_reward - predicted_reward)
        reward_errors.append(reward_error)

        print(
            f"{idx:8d} | "
            f"{window_len:8d} | "
            f"{actual_reward:15.8f} | "
            f"{predicted_reward:15.8f} | "
            f"{reward_error:15.8f}"
        )

    print("=" * 100)
    reward_mae = np.mean(reward_errors)
    reward_rmse = np.sqrt(np.mean(np.square(reward_errors)))

    print(f"Reward MAE: {reward_mae:.8f}, RMSE: {reward_rmse:.8f}")
    print(f"Overall MSE (from file): {data['final_mse']:.8f}")
    print("=" * 100)


def evaluate_interactive(
    r_model: nn.Module,
    env: gym.Env,
    delay: rewdelay.RewardDelay,
    num_episodes: int = 5,
) -> None:
    """
    Run live environment rollouts with delayed rewards and compare predictions.

    Args:
        r_model: Loaded reward model
        env: Gymnasium environment
        delay: Reward delay distribution
        num_episodes: Number of episodes to run (default: 5)
    """
    device = next(r_model.parameters()).device
    all_reward_errors = []

    print("\n" + "=" * 100)
    print(f"Interactive Evaluation - Running {num_episodes} episodes")
    print("=" * 100)

    total_windows = 0

    for episode_idx in range(num_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        window_count = 0

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")
        print("-" * 100)
        print(
            f"{'Window':>8s} | "
            f"{'Length':>8s} | "
            f"{'Actual Reward':>15s} | "
            f"{'Predicted Reward':>15s} | "
            f"{'Error':>15s}"
        )
        print("-" * 100)

        while not (terminated or truncated):
            # Collect current window
            curr_states = []
            curr_actions = []
            curr_terms = []
            curr_actual_rewards = []

            window_delay = delay.sample()
            step_count = 0

            while step_count < window_delay and not (terminated or truncated):
                action = env.action_space.sample()

                curr_states.append(observation)
                curr_actions.append(action)
                curr_terms.append(float(terminated))

                next_observation, reward, terminated, truncated, info = env.step(action)
                curr_actual_rewards.append(reward)

                observation = next_observation
                step_count += 1

            # Only evaluate if we have a full window
            if step_count == window_delay:
                # Compute actual aggregate reward
                actual_aggregate_reward = sum(curr_actual_rewards)

                # Prepare window tensors
                curr_states_tensor = (
                    torch.tensor(curr_states, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .to(device)
                )
                curr_actions_tensor = (
                    torch.tensor(curr_actions, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .to(device)
                )
                curr_terms_tensor = (
                    torch.tensor(
                        [[term_val] for term_val in curr_terms],
                        dtype=torch.float32,
                        device=device,
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                # Predict window reward
                with torch.no_grad():
                    r_preds = r_model(
                        curr_states_tensor, curr_actions_tensor, curr_terms_tensor
                    )
                    predicted_aggregate_reward = r_preds.sum().item()

                # Compute error
                reward_error = abs(actual_aggregate_reward - predicted_aggregate_reward)
                all_reward_errors.append(reward_error)

                # Display first 20 windows
                if window_count < 20:
                    print(
                        f"{window_count:8d} | "
                        f"{step_count:8d} | "
                        f"{actual_aggregate_reward:15.8f} | "
                        f"{predicted_aggregate_reward:15.8f} | "
                        f"{reward_error:15.8f}"
                    )
                elif window_count == 20:
                    print("..." + " " * 96)

                window_count += 1
                total_windows += 1

        print("-" * 100)
        if window_count > 0:
            episode_reward_mae = np.mean(all_reward_errors[-window_count:])
            print(
                f"Episode {episode_idx + 1} Reward MAE: {episode_reward_mae:.8f} ({window_count} windows)"
            )

    print("=" * 100)
    if all_reward_errors:
        reward_mae = np.mean(all_reward_errors)
        reward_rmse = np.sqrt(np.mean(np.square(all_reward_errors)))
        print(
            f"Overall Reward MAE: {reward_mae:.8f}, RMSE: {reward_rmse:.8f} ({total_windows} windows)"
        )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate O2 reward estimation models (MLP only)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model (model_mlp.pt and config.json)",
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

    if args.mode == "predictions":
        # Evaluate from predictions file
        predictions_path = pathlib.Path(args.model_dir) / "predictions_mlp.json"
        evaluate_from_predictions_file(str(predictions_path), args.num_examples)

    elif args.mode == "interactive":
        # Load model
        model_path = pathlib.Path(args.model_dir) / "model_mlp.pt"

        r_model = load_model(
            str(model_path),
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config.get("hidden_dim", 256),
        )
        print(f"Loaded MLP model from {model_path}")

        # Create environment
        env_name = args.env if args.env else config["env_name"]
        env = gym.make(env_name, max_episode_steps=args.max_episode_steps)
        print(f"Created environment: {env_name}")

        # Create delay
        delay = rewdelay.FixedDelay(args.delay)
        print(f"Using fixed delay: {args.delay}")

        # Run interactive evaluation
        evaluate_interactive(r_model, env, delay, args.num_episodes)

        env.close()


if __name__ == "__main__":
    main()
