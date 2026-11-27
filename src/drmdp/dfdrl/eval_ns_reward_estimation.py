"""
Non-stationary reward estimation comparison.

Trains DGRA-style (est_o2 architecture) reward prediction models under two
conditions, each evaluated on held-out data from its own condition:

  (A) Stationary: trained and evaluated on default MuJoCo dynamics.
  (B) Non-stationary: trained and evaluated on dynamics with per-episode
      mass/friction randomisation and per-step gravity drift (envs.py wrapper).

The comparison measures the best achievable reward estimation quality in each
regime — how much harder is reward prediction when dynamics are non-stationary?

Usage:
    python src/drmdp/dfdrl/eval_ns_reward_estimation.py \
        --delay 15 --buffer-num-steps 2000 --train-epochs 100 \
        --output-dir /tmp/ns-comparison

    python src/drmdp/dfdrl/eval_ns_reward_estimation.py \
        --envs HalfCheetah-v5 --delay 15 --buffer-num-steps 3000
"""

import argparse
import dataclasses
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from drmdp import dataproc, envs, rewdelay
from drmdp.dfdrl import est_o2

DEFAULT_ENVS: Sequence[str] = ("HalfCheetah-v5", "Hopper-v5")

# Per-environment defaults sourced from specs/control-local-batch.json (HalfCheetah)
# and specs/bkp.json (Hopper).  DGRA reward_model_kwargs fields are used directly.
ENV_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "HalfCheetah-v5": {
        "max_episode_steps": 1000,
        "delay": 15,
        "batch_size": 64,
        "regu_lam": 1.0,
        "learning_rate": 1e-3,
        "train_epochs": 10,
        # "reward_model_kwargs": {"num_hidden_layers": 5},
    },
    "Hopper-v5": {
        "max_episode_steps": 1000,
        "delay": 15,
        "batch_size": 64,
        "regu_lam": 1.0,
        "learning_rate": 1e-3,
        "train_epochs": 50,
    },
}


@dataclasses.dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for a single environment comparison run."""

    env: str
    max_episode_steps: int
    delay: int
    buffer_num_steps: int
    train_epochs: int
    batch_size: int
    regu_lam: float
    learning_rate: float
    seed: int
    output_dir: str
    reward_model_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def run_comparison(config: ComparisonConfig) -> Mapping[str, Any]:
    """Train and evaluate reward models under stationary and non-stationary dynamics.

    Each model is trained and evaluated within its own condition so the
    comparison measures the best achievable estimation quality in each regime.
    """
    logging.info("=" * 70)
    logging.info(
        "Environment: %s  delay=%d  steps=%d",
        config.env,
        config.delay,
        config.buffer_num_steps,
    )
    logging.info("=" * 70)

    stationary_env = gym.make(config.env, max_episode_steps=config.max_episode_steps)
    nonstationary_env = envs.make_non_stationary_mujoco_env(
        config.env, max_episode_steps=config.max_episode_steps
    )
    obs_dim = int(stationary_env.observation_space.shape[0])
    action_dim = int(stationary_env.action_space.shape[0])

    delay = rewdelay.ClippedPoissonDelay(config.delay, min_delay=2)
    effective_steps = config.buffer_num_steps
    eval_seed = config.seed + 1000

    logging.info("Collecting stationary train data (%d raw steps)...", effective_steps)
    stat_train_windows = _collect_windows(
        stationary_env, delay, effective_steps, seed=config.seed
    )
    logging.info("  %d windows extracted", len(stat_train_windows))

    logging.info("Collecting stationary eval data (seed=%d)...", eval_seed)
    stat_eval_windows = _collect_windows(
        stationary_env, delay, effective_steps, seed=eval_seed
    )
    logging.info("  %d windows extracted", len(stat_eval_windows))

    logging.info(
        "Collecting non-stationary train data (%d raw steps)...", effective_steps
    )
    ns_train_windows = _collect_windows(
        nonstationary_env, delay, effective_steps, seed=config.seed
    )
    logging.info("  %d windows extracted", len(ns_train_windows))

    logging.info("Collecting non-stationary eval data (seed=%d)...", eval_seed)
    ns_eval_windows = _collect_windows(
        nonstationary_env, delay, effective_steps, seed=eval_seed
    )
    logging.info("  %d windows extracted", len(ns_eval_windows))

    stationary_env.close()
    nonstationary_env.close()

    stat_train_ds = _make_dataset(stat_train_windows)
    stat_eval_ds = _make_dataset(stat_eval_windows)
    ns_train_ds = _make_dataset(ns_train_windows)
    ns_eval_ds = _make_dataset(ns_eval_windows)

    logging.info("Training stationary model (%d epochs)...", config.train_epochs)
    stat_model = _train_model(
        obs_dim,
        action_dim,
        stat_train_ds,
        config.train_epochs,
        config.batch_size,
        config.regu_lam,
        config.learning_rate,
        config.seed,
        config.reward_model_kwargs,
    )

    logging.info("Training non-stationary model (%d epochs)...", config.train_epochs)
    ns_model = _train_model(
        obs_dim,
        action_dim,
        ns_train_ds,
        config.train_epochs,
        config.batch_size,
        config.regu_lam,
        config.learning_rate,
        config.seed,
        config.reward_model_kwargs,
    )

    logging.info("Evaluating stationary model on stationary eval set...")
    stat_metrics, stat_preds = _evaluate(
        stat_model, stat_eval_ds, config.batch_size, config.regu_lam
    )
    logging.info("Evaluating non-stationary model on non-stationary eval set...")
    ns_metrics, ns_preds = _evaluate(
        ns_model, ns_eval_ds, config.batch_size, config.regu_lam
    )

    stat_perstep = _per_step_metrics(stat_preds)
    ns_perstep = _per_step_metrics(ns_preds)

    result: Dict[str, Any] = {
        "env": config.env,
        "config": dataclasses.asdict(config),
        "num_windows": {
            "stationary_train": len(stat_train_windows),
            "stationary_eval": len(stat_eval_windows),
            "nonstationary_train": len(ns_train_windows),
            "nonstationary_eval": len(ns_eval_windows),
        },
        "stationary": {
            "window": stat_metrics,
            "per_step": stat_perstep,
        },
        "nonstationary": {
            "window": ns_metrics,
            "per_step": ns_perstep,
        },
    }

    _print_comparison(result)
    _save_results(result, config.output_dir, config.env)
    return result


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _collect_windows(
    env: gym.Env,
    delay: rewdelay.RewardDelay,
    num_steps: int,
    seed: Optional[int] = None,
) -> List[Any]:
    """Collect raw trajectories via random policy, extract delayed-reward windows."""
    buffer = dataproc.collection_traj_data(
        env, steps=num_steps, include_term=True, seed=seed
    )
    windows: List[Any] = est_o2.delayed_reward_data(buffer, delay=delay)
    return windows


def _make_dataset(
    windows: Sequence[Any],
) -> est_o2.DictDataset:
    """Wrap extracted windows into a DictDataset."""
    inputs_list, labels_list = zip(*windows)
    return est_o2.DictDataset(list(inputs_list), list(labels_list))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_model(
    obs_dim: int,
    action_dim: int,
    dataset: data.Dataset,
    train_epochs: int,
    batch_size: int,
    regu_lam: float,
    learning_rate: float,
    seed: int,
    reward_model_kwargs: Optional[Mapping[str, Any]] = None,
) -> est_o2.RNetwork:
    """Train an est_o2 RNetwork and return the trained model."""
    torch.manual_seed(seed)
    model = est_o2.RNetwork(
        state_dim=obs_dim, action_dim=action_dim, **(reward_model_kwargs or {})
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    reward_criterion = nn.MSELoss()
    regu_criterion = nn.MSELoss()

    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=est_o2.collate_variable_length_sequences,
    )

    for epoch_idx in range(train_epochs):
        epoch_losses: List[float] = []
        for inputs, labels in train_loader:
            outputs = model(**inputs)
            window_reward = torch.sum(outputs, dim=1).squeeze(-1)
            aggregate_rewards = labels["aggregate_reward"].float()

            reward_loss = reward_criterion(window_reward, aggregate_rewards)
            regu_loss = regu_criterion(
                labels["start_return"] + window_reward, labels["end_return"]
            )
            loss = reward_loss + regu_lam * regu_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if (epoch_idx + 1) % max(train_epochs // 5, 1) == 0:
            avg_loss = np.mean(epoch_losses)
            logging.info(
                "  Epoch [%d/%d], RMSE: %.6f",
                epoch_idx + 1,
                train_epochs,
                np.sqrt(avg_loss),
            )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    model: est_o2.RNetwork,
    test_ds: data.Dataset,
    batch_size: int,
    regu_lam: float,
) -> Tuple[Mapping[str, Mapping[str, float]], List[Dict[str, Any]]]:
    """Evaluate a model on a test dataset, return window-level and raw predictions."""
    raw_metrics, predictions = est_o2.evaluate_model(
        model, test_ds, batch_size, regu_lam=regu_lam
    )
    mse = {key: float(np.mean(val)) for key, val in raw_metrics.items()}
    rmse = {key: float(np.sqrt(mse[key])) for key in mse}
    return {"mse": mse, "rmse": rmse}, predictions


def _per_step_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> Mapping[str, float]:
    """Compute per-step reward prediction accuracy from evaluation predictions.

    Padded positions (all-zero state vectors from variable-length collation)
    are excluded so they do not dilute the metrics.
    """
    all_true: List[float] = []
    all_pred: List[float] = []

    for pred in predictions:
        states = pred["state"]
        true_steps = pred["per_step_rewards"]
        pred_steps = pred["per_step_predictions"]
        seq_len = states.shape[0]
        for step_idx in range(seq_len):
            if np.any(states[step_idx] != 0.0):
                all_true.append(float(true_steps[step_idx]))
                all_pred.append(float(pred_steps[step_idx]))

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)

    mae = float(np.mean(np.abs(true_arr - pred_arr)))
    mse = float(np.mean((true_arr - pred_arr) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((true_arr - pred_arr) ** 2))
    ss_tot = float(np.sum((true_arr - np.mean(true_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    window_aggs = np.array([float(pred["actual_reward"]) for pred in predictions])

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r_squared,
        "num_steps": len(all_true),
        "step_reward_min": float(true_arr.min()),
        "step_reward_max": float(true_arr.max()),
        "step_reward_mean": float(true_arr.mean()),
        "step_reward_std": float(true_arr.std()),
        "window_agg_min": float(window_aggs.min()),
        "window_agg_max": float(window_aggs.max()),
        "window_agg_mean": float(window_aggs.mean()),
        "window_agg_std": float(window_aggs.std()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_comparison(result: Mapping[str, Any]) -> None:
    """Log a formatted comparison table."""
    env_name = result["env"]
    stat = result["stationary"]
    ns = result["nonstationary"]
    stat_ps = stat["per_step"]
    ns_ps = ns["per_step"]
    counts = result["num_windows"]

    logging.info("")
    logging.info("Results: %s", env_name)
    logging.info(
        "  Windows: stat %d train / %d eval, ns %d train / %d eval",
        counts["stationary_train"],
        counts["stationary_eval"],
        counts["nonstationary_train"],
        counts["nonstationary_eval"],
    )
    fmt6 = "  %-35s  %-14.6f  %-14.6f"
    fmt4 = "  %-35s  %-14.4f  %-14.4f"

    logging.info("")
    logging.info(
        "  %-35s  %-14s  %-14s",
        "Metric",
        "Stationary",
        "Non-stationary",
    )
    logging.info("  %s", "-" * 65)

    logging.info("  Eval reward distribution (per-step)")
    logging.info(fmt6, "  min", stat_ps["step_reward_min"], ns_ps["step_reward_min"])
    logging.info(fmt6, "  max", stat_ps["step_reward_max"], ns_ps["step_reward_max"])
    logging.info(fmt6, "  mean", stat_ps["step_reward_mean"], ns_ps["step_reward_mean"])
    logging.info(fmt6, "  std", stat_ps["step_reward_std"], ns_ps["step_reward_std"])

    logging.info("  Eval reward distribution (window aggregate)")
    logging.info(fmt6, "  min", stat_ps["window_agg_min"], ns_ps["window_agg_min"])
    logging.info(fmt6, "  max", stat_ps["window_agg_max"], ns_ps["window_agg_max"])
    logging.info(fmt6, "  mean", stat_ps["window_agg_mean"], ns_ps["window_agg_mean"])
    logging.info(fmt6, "  std", stat_ps["window_agg_std"], ns_ps["window_agg_std"])

    logging.info("  %s", "-" * 65)
    logging.info("  Estimation quality")
    logging.info(
        fmt6,
        "  Window agg RMSE (reward)",
        stat["window"]["rmse"]["reward"],
        ns["window"]["rmse"]["reward"],
    )
    logging.info(
        fmt6,
        "  Window agg RMSE (regu)",
        stat["window"]["rmse"]["regu"],
        ns["window"]["rmse"]["regu"],
    )
    logging.info(
        fmt6,
        "  Window agg RMSE (total)",
        stat["window"]["rmse"]["total"],
        ns["window"]["rmse"]["total"],
    )
    logging.info(fmt6, "  Per-step MAE", stat_ps["mae"], ns_ps["mae"])
    logging.info(fmt6, "  Per-step RMSE", stat_ps["rmse"], ns_ps["rmse"])
    logging.info(fmt4, "  Per-step R²", stat_ps["r2"], ns_ps["r2"])
    logging.info("  %s", "-" * 65)
    logging.info("")


def _save_results(
    result: Mapping[str, Any],
    output_dir: str,
    env_name: str,
) -> None:
    """Write comparison results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = env_name.replace("-", "_").lower()
    path = os.path.join(output_dir, f"ns_comparison_{safe_name}.json")
    with open(path, "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2)
    logging.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Reward model hyperparameters (delay, batch_size, regu_lam, learning_rate,
    train_epochs) default to per-environment values from ENV_DEFAULTS (sourced
    from specs/control-local-batch.json and specs/bkp.json).  CLI flags
    override them globally when provided.
    """
    parser = argparse.ArgumentParser(
        description="Compare reward estimation under stationary vs non-stationary dynamics"
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=list(DEFAULT_ENVS),
        help="MuJoCo environment IDs to compare (default: HalfCheetah-v5 Hopper-v5)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=None,
        help="Override delay for all envs (default: per-env from spec)",
    )
    parser.add_argument(
        "--buffer-num-steps",
        type=int,
        default=2000,
        help="Base step count for data collection (multiplied by max delay internally)",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Override train epochs for all envs (default: per-env from spec)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for all envs (default: per-env from spec)",
    )
    parser.add_argument(
        "--regu-lam",
        type=float,
        default=None,
        help="Override regu_lam for all envs (default: per-env from spec)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate for all envs (default: per-env from spec)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "ns-comparison"),
    )
    return parser.parse_args()


def _build_config(
    env_id: str,
    args: argparse.Namespace,
) -> ComparisonConfig:
    """Build a ComparisonConfig from per-env defaults with optional CLI overrides."""
    fallback: Dict[str, Any] = {
        "max_episode_steps": 1000,
        "delay": 15,
        "batch_size": 64,
        "regu_lam": 1.0,
        "learning_rate": 1e-3,
        "train_epochs": 10,
    }
    env_cfg = ENV_DEFAULTS.get(env_id, fallback)
    return ComparisonConfig(
        env=env_id,
        max_episode_steps=env_cfg["max_episode_steps"],
        delay=args.delay if args.delay is not None else env_cfg["delay"],
        buffer_num_steps=args.buffer_num_steps,
        train_epochs=(
            args.train_epochs
            if args.train_epochs is not None
            else env_cfg["train_epochs"]
        ),
        batch_size=(
            args.batch_size if args.batch_size is not None else env_cfg["batch_size"]
        ),
        regu_lam=(args.regu_lam if args.regu_lam is not None else env_cfg["regu_lam"]),
        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else env_cfg["learning_rate"]
        ),
        seed=args.seed,
        output_dir=args.output_dir,
        reward_model_kwargs=env_cfg.get("reward_model_kwargs", {}),
    )


def main() -> None:
    """Entry point: run comparison for each requested environment."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    all_results: List[Mapping[str, Any]] = []

    for env_id in args.envs:
        config = _build_config(env_id, args)
        result = run_comparison(config)
        all_results.append(result)

    if len(all_results) > 1:
        logging.info("=" * 70)
        logging.info("SUMMARY ACROSS ENVIRONMENTS")
        logging.info("=" * 70)
        for result in all_results:
            _print_comparison(result)


if __name__ == "__main__":
    main()
