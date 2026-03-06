"""
Evaluation module for O2 (return-grounded reward estimation) models.

Loads saved dual models (reward + return) from est_o2.py and evaluates them.

Evaluation modes:
1. Predictions mode: Loads and displays predictions from saved JSON
2. Interactive mode: Runs live environment rollouts with delayed rewards

Usage Examples:
    # Evaluate from saved predictions (RNN reward model)
    python -m drmdp.dfdrl.eval_est_o2 \
        --model-dir outputs/o2/1709564425 \
        --reward-model-type rnn \
        --mode predictions \
        --num-examples 10

    # Interactive evaluation with live environment
    python -m drmdp.dfdrl.eval_est_o2 \
        --model-dir outputs/o2/1709564425 \
        --reward-model-type rnn \
        --mode interactive \
        --env MountainCarContinuous-v0 \
        --delay 3 \
        --lam 0.5 \
        --xi 0.5 \
        --num-episodes 5
"""

import argparse
import json
import pathlib
import typing
from typing import Any, Dict, List, Tuple

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


def load_dual_models(
    model_path: str,
    reward_model_type: str,
    state_dim: int,
    action_dim: int,
    max_episode_steps: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_heads: int = 2,
    return_model_type: str = "transformer",
) -> Tuple[nn.Module, nn.Module]:
    """
    Load dual models (reward + return) from checkpoint.

    Supports both formats:
    - Two-stage: model_path points to stage2/ directory or parent directory
    - Legacy: model_path points to model_{type}_return.pt file

    Args:
        model_path: Path to model file or directory containing models
        reward_model_type: Type of reward model ("mlp", "rnn", or "transformer")
        state_dim: State dimension
        action_dim: Action dimension
        max_episode_steps: Maximum episode steps (for return model)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of layers (default: 2)
        num_heads: Number of attention heads (default: 2)
        return_model_type: Type of return model (only "transformer" is supported) (default: "transformer")

    Returns:
        Tuple of (reward_model, return_model)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Create reward model
    r_model: nn.Module
    if reward_model_type == "mlp":
        r_model = est_o2.RNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
    elif reward_model_type == "rnn":
        r_model = est_o2.RNetworkRNN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type="lstm",
        )
    elif reward_model_type == "transformer":
        r_model = est_o2.RNetworkTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    else:
        raise ValueError(
            f"Unknown reward_model_type: {reward_model_type}. Use 'mlp', 'rnn', or 'transformer'."
        )

    # Step 2: Create return model
    if return_model_type == "transformer":
        g_model = est_o2.GNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_episode_steps=max_episode_steps,
        )
    else:
        raise ValueError(
            f"Unknown return_model_type: {return_model_type}. Only 'transformer' is supported."
        )

    # Step 3: Load checkpoint - auto-detect format
    model_path_obj = pathlib.Path(model_path)

    # NEW: Check if this is a two-stage checkpoint directory
    if model_path_obj.is_dir():
        # Assume stage2 directory or parent directory
        if model_path_obj.name == "stage2":
            # Direct path to stage2
            dual_model_file = model_path_obj / f"model_{reward_model_type}_return.pt"
        else:
            # Parent directory, look for stage2 subdirectory
            stage2_dir = model_path_obj / "stage2"
            if stage2_dir.exists():
                dual_model_file = stage2_dir / f"model_{reward_model_type}_return.pt"
            else:
                # Fallback: look for legacy file in current directory
                dual_model_file = (
                    model_path_obj / f"model_{reward_model_type}_return.pt"
                )

        if not dual_model_file.exists():
            raise FileNotFoundError(
                f"Two-stage checkpoint not found: {dual_model_file}"
            )
        checkpoint = torch.load(
            dual_model_file, map_location=device, weights_only=False
        )
        print(f"Loaded two-stage checkpoint from: {dual_model_file}")
    else:
        # Legacy single-file checkpoint
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        checkpoint = torch.load(model_path_obj, map_location=device, weights_only=False)
        print(f"Loaded legacy checkpoint from: {model_path_obj}")

    r_model.load_state_dict(checkpoint["reward_model_state_dict"])
    g_model.load_state_dict(checkpoint["return_model_state_dict"])

    r_model.to(device)
    g_model.to(device)
    r_model.eval()
    g_model.eval()

    return r_model, g_model


def evaluate_from_predictions_file(
    predictions_path: str, num_examples: int = 10
) -> None:
    """
    Load and display predictions from saved JSON file.

    Shows dual predictions: rewards + returns for consecutive windows.

    Args:
        predictions_path: Path to predictions_{reward_model_type}_return.json
        num_examples: Number of examples to display (default: 10)
    """
    predictions_path_obj = pathlib.Path(predictions_path)
    if not predictions_path_obj.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(predictions_path_obj, "r", encoding="UTF-8") as readable:
        data = json.load(readable)

    predictions = data["predictions"]
    num_to_show = min(num_examples, len(predictions))

    print("\n" + "=" * 160)
    print(
        f"O2 Dual Model Predictions (Reward: {data['reward_model_type'].upper()}, Return: {data['return_model_type'].upper()})"
    )
    print(f"Showing {num_to_show}/{len(predictions)} examples")
    print("=" * 160)
    print(
        f"{'Window':>8s} | "
        f"{'Prev Len':>8s} | "
        f"{'Curr Len':>8s} | "
        f"{'Act R':>15s} | "
        f"{'Pred R':>15s} | "
        f"{'Err R':>15s} | "
        f"{'Act G_p':>15s} | "
        f"{'Pred G_p':>15s} | "
        f"{'Err G':>15s}"
    )
    print("=" * 160)

    reward_errors = []
    return_errors = []

    for idx in range(num_to_show):
        pred = predictions[idx]
        prev_window = pred["prev_window"]
        curr_window = pred["curr_window"]

        prev_len = len(prev_window["state"])
        curr_len = len(curr_window["state"])

        # Reward prediction (current window aggregate)
        actual_reward = curr_window["actual_aggregate_reward"]
        predicted_reward = curr_window["predicted_aggregate_reward"]
        reward_error = abs(actual_reward - predicted_reward)
        reward_errors.append(reward_error)

        # Return prediction (previous window)
        actual_return_prev = prev_window["actual_return"]
        predicted_return_prev = prev_window["predicted_return"]
        return_error = abs(actual_return_prev - predicted_return_prev)
        return_errors.append(return_error)

        print(
            f"{idx:8d} | "
            f"{prev_len:8d} | "
            f"{curr_len:8d} | "
            f"{actual_reward:15.8f} | "
            f"{predicted_reward:15.8f} | "
            f"{reward_error:15.8f} | "
            f"{actual_return_prev:15.8f} | "
            f"{predicted_return_prev:15.8f} | "
            f"{return_error:15.8f}"
        )

    print("=" * 160)
    reward_mae = np.mean(reward_errors)
    reward_rmse = np.sqrt(np.mean(np.square(reward_errors)))
    return_mae = np.mean(return_errors)
    return_rmse = np.sqrt(np.mean(np.square(return_errors)))

    print(f"Reward MAE: {reward_mae:.8f}, RMSE: {reward_rmse:.8f}")
    print(f"Return MAE: {return_mae:.8f}, RMSE: {return_rmse:.8f}")
    print(f"Overall Reward MSE (from file): {data['final_reward_mse']:.8f}")
    print(f"Overall Return MSE (from file): {data['final_return_mse']:.8f}")
    print("=" * 160)


def evaluate_interactive(
    r_model: nn.Module,
    g_model: nn.Module,
    env: gym.Env,
    delay: rewdelay.RewardDelay,
    num_episodes: int = 5,
    lam: float = 1.0,
    xi: float = 1.0,
) -> None:
    """
    Run live environment rollouts with delayed rewards and compare dual predictions.

    Evaluates both reward predictions (aggregate per window) and return predictions
    (cumulative return at end of window). Computes regularizers ρ₁ and ρ₂.

    Args:
        r_model: Loaded reward model
        g_model: Loaded return model
        env: Gymnasium environment
        delay: Reward delay distribution
        num_episodes: Number of episodes to run (default: 5)
        lam: Weight for ρ₁ regularizer (default: 1.0)
        xi: Weight for ρ₂ regularizer (default: 1.0)
    """
    device = next(r_model.parameters()).device
    all_reward_errors = []
    all_return_errors = []
    all_rho1 = []
    all_rho2 = []

    print("\n" + "=" * 160)
    print(
        f"Interactive Dual Evaluation - Running {num_episodes} episodes (λ={lam}, ξ={xi})"
    )
    print("=" * 160)

    total_windows = 0

    for episode_idx in range(num_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        window_count = 0
        cumulative_return = 0.0

        # Previous window data
        prev_states: List[Any] = []
        prev_actions: List[Any] = []
        prev_terms: List[float] = []
        prev_return = 0.0

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")
        print("-" * 160)
        print(
            f"{'Win':>4s} | "
            f"{'PLen':>4s} | "
            f"{'CLen':>4s} | "
            f"{'ActR':>12s} | "
            f"{'PrdR':>12s} | "
            f"{'ErrR':>12s} | "
            f"{'ActGp':>12s} | "
            f"{'PrdGp':>12s} | "
            f"{'ActGc':>12s} | "
            f"{'PrdGc':>12s} | "
            f"{'ρ₁':>12s} | "
            f"{'ρ₂':>12s}"
        )
        print("-" * 160)

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
                cumulative_return += reward

                observation = next_observation
                step_count += 1

            # Only evaluate if we have a full current window
            if step_count == window_delay:
                # Compute actual values
                actual_aggregate_reward = sum(curr_actual_rewards)
                curr_return = cumulative_return

                # Prepare current window tensors
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

                # Predict current window rewards (r_model)
                with torch.no_grad():
                    r_preds = r_model(
                        curr_states_tensor, curr_actions_tensor, curr_terms_tensor
                    )
                    predicted_aggregate_reward = r_preds.sum().item()

                # Predict current window return (g_model)
                with torch.no_grad():
                    predicted_return_curr = (
                        g_model(
                            curr_states_tensor,
                            curr_actions_tensor,
                            curr_terms_tensor,
                            mask=None,
                            timestep=None,
                        )
                        .squeeze()
                        .item()
                    )

                # For previous window predictions
                if window_count > 0:
                    # Prepare previous window tensors
                    prev_states_tensor = (
                        torch.tensor(prev_states, dtype=torch.float32, device=device)
                        .unsqueeze(0)
                        .to(device)
                    )
                    prev_actions_tensor = (
                        torch.tensor(prev_actions, dtype=torch.float32, device=device)
                        .unsqueeze(0)
                        .to(device)
                    )
                    prev_terms_tensor = (
                        torch.tensor(
                            [[term_val] for term_val in prev_terms],
                            dtype=torch.float32,
                            device=device,
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                    # Predict previous window return (g_model)
                    with torch.no_grad():
                        predicted_return_prev = (
                            g_model(
                                prev_states_tensor,
                                prev_actions_tensor,
                                prev_terms_tensor,
                                mask=None,
                                timestep=None,
                            )
                            .squeeze()
                            .item()
                        )

                    # Compute regularizers
                    # ρ₁: [(Ĝ_curr - R^o_curr) - Ĝ_prev]²
                    rho1 = (
                        (predicted_return_curr - actual_aggregate_reward)
                        - predicted_return_prev
                    ) ** 2

                    # ρ₂: [(G_curr - R̂^o_curr) - G_prev]²
                    rho2 = (
                        (curr_return - predicted_aggregate_reward) - prev_return
                    ) ** 2

                    all_rho1.append(rho1)
                    all_rho2.append(rho2)
                else:
                    # First window: no previous window
                    predicted_return_prev = 0.0
                    rho1 = 0.0
                    rho2 = 0.0

                # Compute errors
                reward_error = abs(actual_aggregate_reward - predicted_aggregate_reward)
                return_error = abs(prev_return - predicted_return_prev)

                all_reward_errors.append(reward_error)
                if (
                    window_count > 0
                ):  # Only count return errors when we have prev window
                    all_return_errors.append(return_error)

                # Display first 20 windows
                if window_count < 20:
                    print(
                        f"{window_count:4d} | "
                        f"{len(prev_states):4d} | "
                        f"{step_count:4d} | "
                        f"{actual_aggregate_reward:12.6f} | "
                        f"{predicted_aggregate_reward:12.6f} | "
                        f"{reward_error:12.6f} | "
                        f"{prev_return:12.6f} | "
                        f"{predicted_return_prev:12.6f} | "
                        f"{curr_return:12.6f} | "
                        f"{predicted_return_curr:12.6f} | "
                        f"{rho1:12.6f} | "
                        f"{rho2:12.6f}"
                    )
                elif window_count == 20:
                    print("..." + " " * 157)

                # Update previous window for next iteration
                prev_states = curr_states
                prev_actions = curr_actions
                prev_terms = curr_terms
                prev_return = curr_return

                window_count += 1
                total_windows += 1

        print("-" * 160)
        if window_count > 0:
            episode_reward_mae = np.mean(all_reward_errors[-window_count:])
            print(
                f"Episode {episode_idx + 1} Reward MAE: {episode_reward_mae:.8f} ({window_count} windows)"
            )

    print("=" * 160)
    if all_reward_errors:
        reward_mae = np.mean(all_reward_errors)
        reward_rmse = np.sqrt(np.mean(np.square(all_reward_errors)))
        print(
            f"Overall Reward MAE: {reward_mae:.8f}, RMSE: {reward_rmse:.8f} ({total_windows} windows)"
        )

    if all_return_errors:
        return_mae = np.mean(all_return_errors)
        return_rmse = np.sqrt(np.mean(np.square(all_return_errors)))
        print(f"Overall Return MAE: {return_mae:.8f}, RMSE: {return_rmse:.8f}")

    if all_rho1:
        avg_rho1 = np.mean(all_rho1)
        avg_rho2 = np.mean(all_rho2)
        print(f"Average ρ₁: {avg_rho1:.8f}")
        print(f"Average ρ₂: {avg_rho2:.8f}")
        combined_loss = (
            np.mean(np.square(all_reward_errors))
            + lam * avg_rho1
            + xi * avg_rho2
            + np.mean(np.square(all_return_errors))
        )
        print(f"Combined loss (approx): {combined_loss:.8f}")

    print("=" * 160)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate O2 (return-grounded reward estimation) models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help=(
            "Directory containing trained models. "
            "For two-stage training, should point to the parent directory "
            "(models will be loaded from stage2/ subdirectory). "
            "For legacy single-stage, should contain model_{type}_return.pt directly."
        ),
    )
    parser.add_argument(
        "--reward-model-type",
        type=str,
        required=True,
        choices=["mlp", "rnn", "transformer"],
        help="Reward model architecture type",
    )
    parser.add_argument(
        "--return-model-type",
        type=str,
        default="transformer",
        choices=["transformer", "rnn"],
        help="Return model architecture type (default: transformer)",
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
        "--lam",
        type=float,
        default=1.0,
        help="Weight for ρ₁ regularizer (default: 1.0)",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=1.0,
        help="Weight for ρ₂ regularizer (default: 1.0)",
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
    print(f"Reward model type: {config['reward_model_type']}")
    print(f"Return model type: {config['return_model_type']}")
    print(f"Environment: {config['env_name']}")

    # Verify model type matches
    if args.reward_model_type != config["reward_model_type"]:
        print(
            f"Warning: Requested reward_model_type '{args.reward_model_type}' differs from config '{config['reward_model_type']}'"
        )
        print(f"Using requested reward_model_type: {args.reward_model_type}")

    if args.mode == "predictions":
        # Evaluate from predictions file
        predictions_path = (
            pathlib.Path(args.model_dir)
            / f"predictions_{args.reward_model_type}_return.json"
        )
        evaluate_from_predictions_file(str(predictions_path), args.num_examples)

    elif args.mode == "interactive":
        # Auto-detect two-stage structure
        model_dir = pathlib.Path(args.model_dir)
        stage2_dir = model_dir / "stage2"

        if stage2_dir.exists():
            print("Detected two-stage training output")
            print(f"Loading models from: {stage2_dir}")
            model_path = stage2_dir
        else:
            print("Using legacy checkpoint structure")
            model_path = model_dir / f"model_{args.reward_model_type}_return.pt"

        max_episode_steps = config["architecture"].get(
            "max_episode_steps", args.max_episode_steps
        )
        r_model, g_model = load_dual_models(
            str(model_path),
            reward_model_type=args.reward_model_type,
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            max_episode_steps=max_episode_steps,
            hidden_dim=config["architecture"].get("hidden_dim", 256),
            num_layers=config["architecture"].get("num_layers", 2),
            num_heads=config["architecture"].get("num_heads", 2),
            return_model_type=args.return_model_type,
        )
        print(
            f"Loaded dual models ({args.reward_model_type.upper()} + {args.return_model_type.upper()}) from {model_path}"
        )
        print(f"Shared embedding verified: {r_model.embedding is g_model.embedding}")

        # Create environment
        env_name = args.env if args.env else config["env_name"]
        env = gym.make(env_name, max_episode_steps=args.max_episode_steps)
        print(f"Created environment: {env_name}")

        # Create delay
        delay = rewdelay.FixedDelay(args.delay)
        print(f"Using fixed delay: {args.delay}")

        # Run interactive evaluation
        evaluate_interactive(
            r_model, g_model, env, delay, args.num_episodes, args.lam, args.xi
        )

        env.close()


if __name__ == "__main__":
    main()
