"""Extract per-step Q^H and Q^C predictions from a trained HC-SAC agent.

Loads an HC model from an experiment output directory, runs deterministic
rollouts, and writes a JSONL file with one entry per step:

    {"episode": 0, "step": 0, "h": -1.23, "c": 2.56, "true_episode_return": -456.78}

Usage:
    python bin/py/eval_hc.py --exp-dir /path/to/output \
        --num-episodes 10 --output /tmp/hc_preds.jsonl --seed 42
"""

import argparse
import collections
import json
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from drmdp import core, envs, rewdelay
from drmdp.control import hc


def _load_config(exp_dir: pathlib.Path) -> Dict[str, Any]:
    with open(exp_dir / "experiment-params.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _make_env(
    config: Mapping[str, Any],
) -> Any:
    env_id = config["env"]
    env_kwargs = config.get("env_kwargs", {})
    non_stationary = config.get("non_stationary", False)
    min_delay = config["min_delay"]
    max_delay = config["max_delay"]

    delay = rewdelay.UniformDelay(min_delay=min_delay, max_delay=max_delay)
    env = envs.make_env(env_id, non_stationary=non_stationary, **env_kwargs)
    env = core.EnvMonitorWrapper(env)
    env = rewdelay.DelayedRewardWrapper(env, delay)
    env = rewdelay.ImputeMissingRewardWrapper(env, impute_value=0.0)
    env = hc.IntervalPositionWrapper(env, max_delay=max_delay)
    return env


def _load_agent(model_path: pathlib.Path, env: Any) -> hc.HCSAC:
    agent = hc.HCSAC.load(str(model_path), env=env)
    agent.policy.set_training_mode(False)
    return agent


def _run_episodes(
    agent: hc.HCSAC,
    env: Any,
    num_episodes: int,
    max_delay: int,
    seed: Optional[int],
) -> List[List[Dict[str, Any]]]:
    device = agent.device
    obs_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.shape[0]
    sa_dim = obs_dim + action_dim

    all_episodes: List[List[Dict[str, Any]]] = []

    for ep_idx in range(num_episodes):
        ep_seed = (seed + ep_idx) if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        history_deque: collections.deque = collections.deque(maxlen=max_delay)
        episode_records: List[Dict[str, Any]] = []
        step_idx = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            history_window = np.zeros((max_delay, sa_dim), dtype=np.float32)
            recent = list(history_deque)
            n_recent = len(recent)
            if n_recent > 0:
                history_window[-n_recent:] = np.stack(recent)
            history_tensor = torch.as_tensor(
                history_window, dtype=torch.float32, device=device
            ).unsqueeze(0)

            action, _ = agent.predict(obs, deterministic=True)

            with torch.no_grad():
                _, h_n = agent.history_encoder(history_tensor)
                h_t = h_n.squeeze(0)
                qh = agent.head_net(h_t)
                qh_val = qh.item()

                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action_tensor = torch.as_tensor(
                    action, dtype=torch.float32, device=device
                ).unsqueeze(0)
                qc_values = agent.critic(obs_tensor, action_tensor)
                qc_cat = torch.cat(qc_values, dim=1)
                qc_min, _ = torch.min(qc_cat, dim=1, keepdim=True)
                qc_val = qc_min.item()

            next_obs, _, terminated, truncated, info = env.step(action)

            episode_records.append(
                {
                    "episode": ep_idx,
                    "step": step_idx,
                    "h": qh_val,
                    "c": qc_val,
                    "true_episode_return": 0.0,
                }
            )

            interval_end = info.get("interval_end", False)
            if interval_end or terminated or truncated:
                history_deque.clear()
            else:
                sa = np.concatenate(
                    [
                        np.asarray(obs, dtype=np.float32).flatten(),
                        np.asarray(action, dtype=np.float32).flatten(),
                    ]
                )
                history_deque.append(sa)

            obs = next_obs
            step_idx += 1

        true_return = info.get("true_episode_return", 0.0)
        for record in episode_records:
            record["true_episode_return"] = true_return

        all_episodes.append(episode_records)

    return all_episodes


def _write_jsonl(
    episodes: Sequence[Sequence[Mapping[str, Any]]], output_path: pathlib.Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for episode in episodes:
            for record in episode:
                fh.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Q^H and Q^C predictions from a trained HC-SAC agent."
    )
    parser.add_argument(
        "--exp-dir",
        type=pathlib.Path,
        required=True,
        help="Experiment output directory (with experiment-params.json and hc_model.zip).",
    )
    parser.add_argument(
        "--num-episodes", type=int, required=True, help="Number of rollout episodes."
    )
    parser.add_argument(
        "--output", type=pathlib.Path, required=True, help="Output JSONL file path."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for environment resets."
    )
    args = parser.parse_args()

    config = _load_config(args.exp_dir)
    max_delay: int = config["max_delay"]

    env = _make_env(config)
    agent = _load_agent(args.exp_dir / "hc_model", env)

    episodes = _run_episodes(agent, env, args.num_episodes, max_delay, args.seed)
    _write_jsonl(episodes, args.output)

    total_steps = sum(len(ep) for ep in episodes)
    print(f"Wrote {total_steps} steps across {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
