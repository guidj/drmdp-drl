"""
Off-policy SAC control with interleaved reward model learning.

The environment rewards are delayed and aggregated; a reward model
(e.g. IRCRRewardModel) estimates per-step rewards that are substituted
into the SAC replay buffer at sample time via RelabelingReplayBuffer.
SAC therefore trains on per-step predictions rather than delayed aggregates.

Usage:
    python src/drmdp/control/runner.py --env MountainCarContinuous-v0 \\
        --delay 3 --num-steps 50000 --reward-model-type ircr \\
        --update-every-n-steps 1000 --output-dir /tmp/control-ircr
"""

import argparse
import ast
import dataclasses
import logging
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3.common import callbacks

from drmdp import logger, rewdelay
from drmdp.control import base, ircr


@dataclasses.dataclass(frozen=True)
class TrainingArgs:
    """Configuration for the off-policy control loop.

    Attributes:
        env: Gymnasium environment name.
        delay: Fixed reward delay (number of steps).
        max_episode_steps: Maximum steps per episode before truncation.
            None uses the environment's default.
        reward_model_type: Reward model identifier. Currently supports "ircr".
        update_every_n_steps: Call reward_model.update() every N env steps.
        clear_buffer_on_update: Reset SAC's replay buffer after each reward
            model update. Useful for methods with dramatic reward redistribution
            (e.g. RUDDER).
        reward_model_kwargs: Keyword arguments forwarded to the reward model
            constructor. Keys and value types depend on the chosen model.
        num_steps: Total environment steps to train for.
        sac_learning_rate: Learning rate for SAC actor and critic networks.
        sac_buffer_size: Capacity of SAC's replay buffer.
        sac_batch_size: Mini-batch size for SAC gradient updates.
        sac_gradient_steps: Gradient steps per env step (-1 = match collected).
        log_episode_frequency: Log to ExperimentLogger every N episodes.
        output_dir: Directory for logs and the saved SAC model.
        seed: Random seed for reproducibility.
    """

    env: str
    delay: int
    max_episode_steps: Optional[int]
    reward_model_type: str
    update_every_n_steps: int
    clear_buffer_on_update: bool
    reward_model_kwargs: Mapping[str, Any]
    num_steps: int
    sac_learning_rate: float
    sac_buffer_size: int
    sac_batch_size: int
    sac_gradient_steps: int
    log_episode_frequency: int
    output_dir: str
    seed: Optional[int] = None


class RewardModelUpdateCallback(callbacks.BaseCallback):
    """SB3 callback that builds episode trajectories and updates the reward model.

    Builds trajectories step-by-step from SB3 callback locals, appends
    completed episodes to a pending buffer, and flushes that buffer to
    the reward model every `update_every_n_steps` env steps.  Episode
    returns and reward model metrics are recorded via ExperimentLogger.
    """

    def __init__(
        self,
        reward_model: base.RewardModel,
        update_every_n_steps: int,
        clear_buffer_on_update: bool,
        log_episode_frequency: int,
        exp_logger: logger.ExperimentLogger,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._reward_model = reward_model
        self._update_every_n_steps = update_every_n_steps
        self._clear_buffer_on_update = clear_buffer_on_update
        self._log_episode_frequency = log_episode_frequency
        self._exp_logger = exp_logger

        # Per-episode accumulators; reset at each episode boundary.
        self._episode_obs: List[np.ndarray] = []
        self._episode_actions: List[np.ndarray] = []
        self._episode_rewards: List[float] = []
        self._episode_terminals: List[bool] = []
        self._episode_steps: int = 0
        self._episode_count: int = 0

        # Completed trajectories not yet passed to the reward model.
        self._pending_trajectories: List[base.Trajectory] = []
        self._last_model_metrics: Mapping[str, float] = {}

    def _on_step(self) -> bool:
        # model._last_obs has shape (n_envs, obs_dim); index 0 for single env.
        obs_before = self.model._last_obs[0]
        action = self.locals["actions"][0]
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self._episode_obs.append(obs_before.copy())
        self._episode_actions.append(action.copy())
        self._episode_rewards.append(reward)
        self._episode_terminals.append(done)
        self._episode_steps += 1

        if done:
            self._on_episode_end()

        if (
            self.num_timesteps % self._update_every_n_steps == 0
            and len(self._pending_trajectories) > 0
        ):
            self._flush_pending_trajectories()

        return True

    def _on_training_end(self) -> None:
        # Incorporate trajectories collected after the last interval update.
        if len(self._pending_trajectories) > 0:
            self._flush_pending_trajectories()

    def _on_episode_end(self) -> None:
        """Finalise the current episode and log if on schedule."""
        trajectory = base.Trajectory(
            observations=np.stack(self._episode_obs),
            actions=np.stack(self._episode_actions),
            env_rewards=np.array(self._episode_rewards, dtype=np.float32),
            terminals=np.array(self._episode_terminals),
            episode_return=float(sum(self._episode_rewards)),
        )
        self._pending_trajectories.append(trajectory)
        self._episode_count += 1

        if self._episode_count % self._log_episode_frequency == 0:
            est_rewards = self._reward_model.predict(
                trajectory.observations,
                trajectory.actions,
                trajectory.terminals,
            )
            self._exp_logger.log(
                episode=self._episode_count,
                steps=self._episode_steps,
                returns=trajectory.episode_return,
                info={
                    "estimated_return": float(est_rewards.sum()),
                    "reward_model": dict(self._last_model_metrics),
                },
            )

        self._episode_obs = []
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_terminals = []
        self._episode_steps = 0

    def _flush_pending_trajectories(self) -> None:
        """Pass pending trajectories to the reward model and optionally clear the buffer."""
        self._last_model_metrics = self._reward_model.update(self._pending_trajectories)
        self._pending_trajectories = []
        if self._clear_buffer_on_update:
            self.model.replay_buffer.reset()


def run(args: TrainingArgs) -> None:
    """Build environment, reward model, and SAC; train to completion.

    The environment is wrapped with DelayedRewardWrapper (to aggregate rewards
    over windows) and ImputeMissingRewardWrapper (to replace None rewards with
    0.0 so SAC always receives a numeric value).
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Training args: %s", args)

    env = gym.make(args.env, max_episode_steps=args.max_episode_steps)
    env = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(args.delay))
    env = rewdelay.ImputeMissingRewardWrapper(env, impute_value=0.0)

    reward_model = _make_reward_model(args)

    with logger.ExperimentLogger(args.output_dir, params=args) as exp_logger:
        sac = stable_baselines3.SAC(
            "MlpPolicy",
            env,
            learning_rate=args.sac_learning_rate,
            buffer_size=args.sac_buffer_size,
            batch_size=args.sac_batch_size,
            gradient_steps=args.sac_gradient_steps,
            replay_buffer_class=base.RelabelingReplayBuffer,
            replay_buffer_kwargs={"reward_model": reward_model},
            seed=args.seed,
            verbose=1,
        )
        callback = RewardModelUpdateCallback(
            reward_model=reward_model,
            update_every_n_steps=args.update_every_n_steps,
            clear_buffer_on_update=args.clear_buffer_on_update,
            log_episode_frequency=args.log_episode_frequency,
            exp_logger=exp_logger,
        )
        sac.learn(total_timesteps=args.num_steps, callback=callback)

    sac.save(os.path.join(args.output_dir, "sac_model"))
    logging.info("Model saved to %s/sac_model", args.output_dir)


def _make_reward_model(args: TrainingArgs) -> base.RewardModel:
    """Instantiate the reward model specified in args."""
    if args.reward_model_type == "ircr":
        return ircr.IRCRRewardModel(**args.reward_model_kwargs)
    raise ValueError(f"Unknown reward_model_type: {args.reward_model_type!r}")


def main() -> None:
    """Parse command-line arguments and launch the control loop."""
    args = parse_args()
    run(args)


def parse_args() -> TrainingArgs:
    """Parse command-line arguments into a TrainingArgs instance."""
    parser = argparse.ArgumentParser(
        description="Off-policy SAC control with interleaved reward model learning"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MountainCarContinuous-v0",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Fixed reward delay (number of steps)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (None = environment default)",
    )
    parser.add_argument(
        "--reward-model-type",
        type=str,
        default="ircr",
        choices=["ircr"],
        help="Reward model to use",
    )
    parser.add_argument(
        "--update-every-n-steps",
        type=int,
        default=1000,
        help="Update the reward model every N environment steps",
    )
    parser.add_argument(
        "--clear-buffer-on-update",
        action="store_true",
        default=False,
        help="Reset SAC replay buffer after each reward model update",
    )
    parser.add_argument(
        "--reward-model-kwarg",
        action="append",
        dest="reward_model_kwargs",
        default=[],
        metavar="KEY=VALUE",
        help="Reward-model-specific keyword argument (repeatable). "
        "Values are parsed via ast.literal_eval; unrecognised literals "
        "are kept as strings. E.g. --reward-model-kwarg max_buffer_size=200",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50000,
        help="Total environment steps to train for",
    )
    parser.add_argument(
        "--sac-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for SAC actor and critic",
    )
    parser.add_argument(
        "--sac-buffer-size",
        type=int,
        default=100000,
        help="Capacity of SAC's replay buffer",
    )
    parser.add_argument(
        "--sac-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for SAC gradient updates",
    )
    parser.add_argument(
        "--sac-gradient-steps",
        type=int,
        default=-1,
        help="Gradient steps per env step (-1 = match collected steps)",
    )
    parser.add_argument(
        "--log-episode-frequency",
        type=int,
        default=10,
        help="Log to ExperimentLogger every N episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=tempfile.gettempdir(),
        help="Directory for logs and saved model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args, _ = parser.parse_known_args()
    args_dict = vars(args)
    args_dict["reward_model_kwargs"] = parse_reward_model_kwargs(
        args_dict["reward_model_kwargs"]
    )
    return TrainingArgs(**args_dict)


def parse_reward_model_kwargs(
    pairs: Sequence[str],
) -> Mapping[str, Any]:
    """Convert a list of 'key=value' strings into a keyword-argument mapping.

    Values are parsed with ast.literal_eval so that integers, floats, and
    booleans are returned with their native types.  Strings that cannot be
    parsed as literals are kept as-is.
    """
    kwargs: Dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            value: Any = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        kwargs[key] = value
    return kwargs


if __name__ == "__main__":
    main()
