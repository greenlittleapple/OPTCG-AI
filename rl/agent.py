from __future__ import annotations

from dataclasses import asdict
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from pettingzoo.utils import conversions
import supersuit as ss

from pettingzoo.utils import BaseWrapper
from env import OPTCGEnv, OPTCGPlayerObs


class SB3ActionMaskWrapper(BaseWrapper, gym.Env):
    """Wrapper to adapt PettingZoo AEC environments for SB3 with action masking."""

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self.observation_space = super().observation_space(self.possible_agents[0])
        self.action_space = super().action_space(self.possible_agents[0])
        return self.observe(self.agent_selection), {}

    def step(self, action: int):
        current_agent = self.agent_selection
        super().step(action)
        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self.rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent: str):
        obs = super().observe(agent)
        if isinstance(obs, OPTCGPlayerObs):
            return asdict(obs)
        return obs

    def action_mask(self):
        return self.env.action_mask(self.env.agent_selection)


class PPOAgent:
    """Minimal PPO agent using Stable-Baselines3."""

    def __init__(self, env: OPTCGEnv, *, verbose: int = 2) -> None:
        env = SB3ActionMaskWrapper(env)
        env.reset()
        env = ActionMasker(env, lambda e: e.action_mask())
        check_env(env)

        self.model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=verbose,
            tensorboard_log="./logs/",
        )

    def train(self, timesteps: int = 1_000) -> None:
        """Train the agent for ``timesteps`` steps."""

        class TrainLogger(BaseCallback):
            # def _on_step(self) -> bool:
            #     for info in self.locals.get("infos", []):
            #         if "episode" in info:
            #             ep = info["episode"]
            #             print(
            #                 f"step={self.num_timesteps} episode_reward={ep['r']} length={ep['l']}"
            #             )
            #     return True

            def _on_rollout_end(self) -> None:
                self.model.logger.dump(self.num_timesteps)

        self.model.learn(total_timesteps=timesteps, progress_bar=True)

    def act(self, obs: OPTCGPlayerObs | dict[str, Any]) -> int:
        """Return the greedy action for *obs* using the trained policy."""
        if not isinstance(obs, dict):
            obs = asdict(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def main(timesteps: int = 50000) -> None:
    env = OPTCGEnv()
    agent = PPOAgent(env)
    agent.train(timesteps)
    print(f'Predicted Action for Player 1: {agent.act(env.observe("player_0"))}')
    print(f'Predicted Action for Player 2: {agent.act(env.observe("player_1"))}')


if __name__ == "__main__":  # pragma: no cover
    main()
