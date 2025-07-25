from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import OPTCGEnv


@dataclass
class DQNConfig:
    """Hyperparameters for :class:`DQNAgent`."""

    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 32
    target_update: int = 10  # episodes


class QNetwork(nn.Module):
    """Simple feed-forward network for DQN."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x.float())


class DQNAgent:
    """Deep Q-Network agent for :class:`OPTCGEnv`."""

    def __init__(
        self,
        env: OPTCGEnv,
        config: DQNConfig | None = None,
        *,
        hooks: List[Callable[["DQNAgent"], None]] | None = None,
    ) -> None:
        self.env = env
        self.config = config or DQNConfig()
        self.hooks = hooks or []

        obs_dim = 3
        action_dim = self.env.action_space(self.env.possible_agents[0]).n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.config.buffer_size)
        self.epsilon = self.config.epsilon_start
        self.steps_done = 0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _obs_to_array(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            ca = obs.get("can_attack", 0)
            cr = obs.get("can_resolve", 0)
            ce = obs.get("can_end_turn", 0)
        else:
            ca = int(getattr(obs, "can_attack", 0))
            cr = int(getattr(obs, "can_resolve", 0))
            ce = int(getattr(obs, "can_end_turn", 0))
        return np.array([ca, cr, ce], dtype=np.float32)

    def _select_action(self, obs: Any, agent: str, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return int(self.env.action_space(agent).sample())

        state = torch.from_numpy(self._obs_to_array(obs)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(torch.argmax(q_values).item())

    def _store(self, obs: Any, action: int, reward: float, next_obs: Any, done: bool) -> None:
        state = self._obs_to_array(obs)
        next_state = self._obs_to_array(next_obs)
        self.memory.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _optimize_model(self) -> None:
        if len(self.memory) < self.config.batch_size:
            return
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        targets = rewards + self.config.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, episodes: int = 100) -> None:
        for ep in range(1, episodes + 1):
            self.env.reset()
            for agent in self.env.agent_iter():
                obs, _, terminated, truncated, _ = self.env.last()
                done = terminated or truncated
                if done:
                    self.env.step(None)
                    continue

                action = self._select_action(obs, agent)
                self.env.step(action)
                next_obs = self.env.observe(agent)

                reward = self.env.rewards.get(agent, 0.0)
                done = self.env.terminations[agent] or self.env.truncations[agent]
                self._store(obs, action, reward, next_obs, done)
                self._optimize_model()

                for hook in self.hooks:
                    hook(self)

            if ep % self.config.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def act(self, obs: Any, agent: str) -> int:
        """Return the greedy action for *obs* (no exploration)."""
        return self._select_action(obs, agent, training=False)


def main(episodes: int = 5) -> None:
    """Run a short training session for manual testing."""
    env = OPTCGEnv()
    agent = DQNAgent(env)
    agent.train(episodes)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    main()

