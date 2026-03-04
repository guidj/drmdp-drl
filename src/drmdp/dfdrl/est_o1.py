"""
Reward prediction networks for delayed, aggregate, and anonymous feedback (DAAF).

This module provides three architectures for predicting rewards from sequences
of (state, action) pairs:

1. RNetwork: Feedforward MLP - processes each (s,a) pair independently
2. RNetworkRNN: Recurrent network - processes sequences with LSTM/GRU
3. RNetworkTransformer: Transformer decoder - processes sequences with self-attention

All models support two modes:
- forward(): Sequential prediction for training on delayed reward sequences
- forward_markovian(): Markovian prediction for test time (reward depends only on current s,a)

Usage Example:
    # Feedforward (no stub needed - each step is independent)
    model = RNetwork(state_dim=4, action_dim=2, hidden_dim=256)
    # Sequential input: (batch_size, seq_len, state_dim), (batch_size, seq_len, action_dim)
    rewards = model(states, actions)  # Output: (batch_size, seq_len, 1)
    # Markovian input: (batch_size, state_dim), (batch_size, action_dim)
    reward = model.forward_markovian(state, action)  # Output: (batch_size, 1)

    # RNN-based (stub handled internally)
    model = RNetworkRNN(state_dim=4, action_dim=2, hidden_dim=256,
                        rnn_type="lstm", num_layers=2)
    rewards = model(states, actions)  # Sequential
    reward = model.forward_markovian(state, action)  # Markovian

    # Transformer-based (stub handled internally)
    model = RNetworkTransformer(state_dim=4, action_dim=2, hidden_dim=256,
                                num_heads=8, num_layers=4)
    rewards = model(states, actions)  # Sequential
    reward = model.forward_markovian(state, action)  # Markovian

All models output per-step reward predictions that are summed during training
to match the delayed aggregate reward signal. RNN and Transformer models
internally prepend a zero stub [0, 0, ...] to sequences for consistent
initialization, ensuring predictions depend only on the current context.
"""

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from drmdp import dataproc, rewdelay


class RNetwork(nn.Module):
    """
    Feedforward MLP for reward prediction.
    Processes (state, action, done) tuples independently.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # +1 for done flag
        self.fc1 = nn.Linear(state_dim + action_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, done):
        features = torch.concat([state, action, done], dim=-1)
        features = nn.functional.relu(self.fc1(features))
        features = nn.functional.relu(self.fc2(features))
        reward = self.fc3(features)
        # TODO: distributional reward prediction
        return reward


class RNetworkRNN(nn.Module):
    """
    RNN-based reward prediction network.
    Processes sequential (state, action, done) tuples and outputs one reward value per step.

    Input shape: (batch_size, sequence_length, state_dim + action_dim + 1)
    Output shape: (batch_size, sequence_length, 1)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_layers=2,
        rnn_type="lstm",
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # Input projection: (state, action, done) -> hidden_dim
        self.input_proj = nn.Linear(state_dim + action_dim + 1, hidden_dim)

        # RNN layer
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}. Use 'lstm' or 'gru'.")

        # Output projection: hidden_dim -> 1 (reward per step)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, done):
        """
        Args:
            state: Tensor of shape (batch_size, sequence_length, state_dim)
            action: Tensor of shape (batch_size, sequence_length, action_dim)
            done: Tensor of shape (batch_size, sequence_length, 1)

        Returns:
            rewards: Tensor of shape (batch_size, sequence_length, 1)
                Predictions for real timesteps only (stub is handled internally)
        """
        batch_size = state.shape[0]
        device = state.device

        # Prepend zero stub for position 0 to ensure consistent RNN initialization
        zero_state = torch.zeros(batch_size, 1, self.state_dim, device=device)
        zero_action = torch.zeros(batch_size, 1, self.action_dim, device=device)
        zero_done = torch.zeros(batch_size, 1, 1, device=device)

        # Create sequence with stub: [zero_stub, actual_sequence]
        state_with_stub = torch.cat([zero_state, state], dim=1)
        action_with_stub = torch.cat([zero_action, action], dim=1)
        done_with_stub = torch.cat([zero_done, done], dim=1)

        # Concatenate state, action, and done along feature dimension
        # Shape: (batch_size, sequence_length+1, state_dim + action_dim + 1)
        features = torch.concat([state_with_stub, action_with_stub, done_with_stub], dim=-1)

        # Project to hidden dimension and apply activation
        # Shape: (batch_size, sequence_length+1, hidden_dim)
        features = nn.functional.relu(self.input_proj(features))

        # Process through RNN
        # rnn_out shape: (batch_size, sequence_length+1, hidden_dim)
        rnn_out, _ = self.rnn(features)

        # Project to reward predictions (one per timestep)
        # Shape: (batch_size, sequence_length+1, 1)
        rewards = self.output_proj(rnn_out)

        # Return only predictions for real timesteps (skip position 0)
        return rewards[:, 1:, :]


class RNetworkTransformer(nn.Module):
    """
    Transformer Decoder-based reward prediction network.
    Processes sequential (state, action, done) tuples and outputs one reward value per step.

    Uses causal masking to ensure predictions at timestep t only depend on steps <= t.

    Input shape: (batch_size, sequence_length, state_dim + action_dim + 1)
    Output shape: (batch_size, sequence_length, 1)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        feedforward_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input embedding: (state, action, done) -> hidden_dim
        self.input_embedding = nn.Linear(state_dim + action_dim + 1, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 1000, hidden_dim)
        )  # Max sequence length 1000

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection: hidden_dim -> 1 (reward per step)
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize positional encoding with sinusoidal pattern
        self._init_positional_encoding()

    def _init_positional_encoding(self):
        """Initialize positional encoding with sinusoidal pattern."""
        max_len = self.pos_encoding.size(1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.hidden_dim)
        )

        pe = torch.zeros(1, max_len, self.hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pe)

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for transformer decoder.
        Ensures that position i can only attend to positions <= i.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, state, action, done):
        """
        Args:
            state: Tensor of shape (batch_size, sequence_length, state_dim)
            action: Tensor of shape (batch_size, sequence_length, action_dim)
            done: Tensor of shape (batch_size, sequence_length, 1)

        Returns:
            rewards: Tensor of shape (batch_size, sequence_length, 1)
                Predictions for real timesteps only (stub is handled internally)
        """
        batch_size = state.shape[0]
        device = state.device

        # Prepend zero stub for position 0 to ensure consistent initialization
        zero_state = torch.zeros(batch_size, 1, self.state_dim, device=device)
        zero_action = torch.zeros(batch_size, 1, self.action_dim, device=device)
        zero_done = torch.zeros(batch_size, 1, 1, device=device)

        # Create sequence with stub: [zero_stub, actual_sequence]
        state_with_stub = torch.cat([zero_state, state], dim=1)
        action_with_stub = torch.cat([zero_action, action], dim=1)
        done_with_stub = torch.cat([zero_done, done], dim=1)

        # Concatenate state, action, and done along feature dimension
        # Shape: (batch_size, sequence_length+1, state_dim + action_dim + 1)
        features = torch.concat([state_with_stub, action_with_stub, done_with_stub], dim=-1)

        seq_len = features.shape[1]

        # Embed input
        # Shape: (batch_size, sequence_length+1, hidden_dim)
        features = self.input_embedding(features)

        # Add positional encoding
        features = features + self.pos_encoding[:, :seq_len, :]

        # Apply layer normalization
        features = self.layer_norm(features)

        # Generate causal mask
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(features.device)

        # For transformer decoder, we use the input as both memory and target
        # This is a decoder-only architecture (similar to GPT)
        # The causal mask ensures autoregressive behavior
        output = self.transformer_decoder(
            tgt=features,
            memory=features,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )

        # Project to reward predictions (one per timestep)
        # Shape: (batch_size, sequence_length+1, 1)
        rewards = self.output_proj(output)

        # Return only predictions for real timesteps (skip position 0)
        return rewards[:, 1:, :]


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

    def create_traj_step(state, action, reward, done):
        return {
            "state": torch.tensor(state),
            "action": torch.tensor(action),
            "done": torch.tensor([float(done)]),
        }, torch.tensor(reward, dtype=torch.float32)

    def create_example(traj_steps: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        # assumes labels single value tensors
        inputs, labels = zip(*traj_steps)
        # agg rewards
        return data.default_collate(inputs), torch.sum(torch.stack(labels))

    states = np.concatenate(
        [
            np.stack([example[0] for example in buffer]),
            np.stack([example[2] for example in buffer]),
        ],
        axis=1,
    )
    action = np.stack([example[1] for example in buffer])
    reward = np.stack([example[3] for example in buffer])
    done = np.stack([example[4] for example in buffer])
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
                states[idx][:obs_dim], action[idx], reward[idx], done[idx]
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

    For each sequence, predicts reward for each (state, action, done) tuple independently,
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

            # Predict reward for each (state, action, done) tuple independently
            per_step_predictions = []
            for step_idx in range(seq_len):
                # Extract (state, action, done) at timestep step_idx for all sequences in batch
                state_t = inputs["state"][:, step_idx, :]  # (batch_size, state_dim)
                action_t = inputs["action"][:, step_idx, :]  # (batch_size, action_dim)
                done_t = inputs["done"][:, step_idx, :]  # (batch_size, 1)

                # Add sequence dimension for RNN/Transformer models
                state_seq = state_t.unsqueeze(1)  # (batch_size, 1, state_dim)
                action_seq = action_t.unsqueeze(1)  # (batch_size, 1, action_dim)
                done_seq = done_t.unsqueeze(1)  # (batch_size, 1, 1)

                # Forward pass - returns (batch_size, 1, 1) for RNN/Transformer, (batch_size, 1) for MLP
                reward_t = model(state_seq, action_seq, done_seq)
                if reward_t.dim() == 3:
                    reward_t = reward_t.squeeze(1)  # (batch_size, 1)
                per_step_predictions.append(reward_t)

            # Stack predictions: (batch_size, seq_len, 1)
            predictions = torch.stack(per_step_predictions, dim=1)

            # Sum predictions across sequence to get aggregate reward
            pred_window_reward = torch.squeeze(torch.sum(predictions, dim=1))
            mean_squared_error = eval_criterion(pred_window_reward, labels)
            errors.append(mean_squared_error)

            # Optionally collect predictions for analysis
            if collect_predictions:
                for idx in range(batch_size_actual):
                    predictions_list.append(
                        {
                            "state": inputs["state"][idx].cpu().numpy(),
                            "action": inputs["action"][idx].cpu().numpy(),
                            "done": inputs["done"][idx].cpu().numpy(),
                            "actual_reward": labels[idx].item(),
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
    batch_size: int,
    eval_steps: int,
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
    elif model_type == "rnn":
        model = RNetworkRNN(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=256,
            num_layers=2,
            rnn_type="lstm",
        )
    elif model_type == "transformer":
        model = RNetworkTransformer(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Use 'mlp', 'rnn', or 'transformer'."
        )

    print(f"Training with {model_type.upper()} model")
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
            # Forward pass (models handle zero stub internally for RNN/Transformer)
            outputs = model(**inputs)

            # Calculate loss for each seq in batch
            # outputs shape: (batch_size, seq_len, 1)
            window_reward = torch.squeeze(torch.sum(outputs, dim=1))
            loss = criterion(window_reward, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        # Evaluation using Markovian predictions
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_file = output_path / f"predictions_{model_type}.json"
    with open(predictions_file, "w") as writable:
        json.dump(
            {
                "model_type": model_type,
                "final_mse": final_mse,
                "num_predictions": len(predictions_list),
                "predictions": [
                    {
                        "state": pred["state"].tolist(),
                        "action": pred["action"].tolist(),
                        "done": pred["done"].tolist(),
                        "actual_reward": pred["actual_reward"],
                        "predicted_reward": pred["predicted_reward"],
                        "per_step_predictions": pred["per_step_predictions"].tolist(),
                    }
                    for pred in predictions_list
                ],
            },
            writable,
            indent=2,
        )
    print(f"Predictions saved to {predictions_file}")

    # Save training metrics
    metrics_file = output_path / f"metrics_{model_type}.json"
    with open(metrics_file, "w") as writable:
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
    model_file = output_path / f"model_{model_type}.pt"
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

    return final_mse, predictions_list


def main():
    parser = argparse.ArgumentParser(
        description="Train reward prediction models for delayed feedback"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rnn",
        choices=["mlp", "rnn", "transformer", "all"],
        help="Model architecture to train (default: mlp). Use 'all' to train all models.",
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
        "--delay",
        type=int,
        default=3,
        help="Fixed delay for reward feedback (default: 3)",
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
    dataset = DictDataset(data.default_collate(inputs), torch.stack(labels))

    # Train model(s)
    if args.model_type == "all":
        print("\n" + "=" * 80)
        print("Training all model types...")
        print("=" * 80)
        results = {}
        for model_type in ["mlp", "rnn", "transformer"]:
            print(f"\n{'=' * 80}")
            print(f"Training {model_type.upper()} model")
            print("=" * 80)
            final_mse, _ = train(
                env,
                dataset=dataset,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
                model_type=model_type,
                output_dir=args.output_dir,
            )
            results[model_type] = final_mse

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        for model_type, mse in results.items():
            rmse = np.sqrt(mse)
            print(f"{model_type.upper():15s}: RMSE = {rmse:.8f}")
    else:
        train(
            env,
            dataset=dataset,
            batch_size=args.batch_size,
            eval_steps=args.eval_steps,
            model_type=args.model_type,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
