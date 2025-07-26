from __future__ import annotations

from dataclasses import asdict
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.utils import conversions
import supersuit as ss

from .env import OPTCGEnv, OPTCGPlayerObs


class PPOAgent:
    """Minimal PPO agent using Stable-Baselines3."""

    def __init__(self, env: OPTCGEnv, *, verbose: int = 1) -> None:
        # Convert AECEnv -> ParallelEnv -> VectorEnv for SB3
        parallel_env = conversions.aec_to_parallel(env)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
        self.vec_env = ss.concat_vec_envs_v1(
            vec_env, 1, num_cpus=1, base_class="stable_baselines3"
        )
        self.vec_env = VecMonitor(self.vec_env)
        self.model = PPO("MultiInputPolicy", self.vec_env, verbose=verbose)

    def train(self, timesteps: int = 1_000) -> None:
        """Train the agent for ``timesteps`` steps."""

        class TrainLogger(BaseCallback):
            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        ep = info["episode"]
                        print(
                            f"step={self.num_timesteps} episode_reward={ep['r']} length={ep['l']}"
                        )
                return True

            def _on_rollout_end(self) -> None:
                self.model.logger.dump(self.num_timesteps)

        self.model.learn(total_timesteps=timesteps, callback=TrainLogger())

    def act(self, obs: OPTCGPlayerObs | dict[str, Any]) -> int:
        """Return the greedy action for *obs* using the trained policy."""
        if not isinstance(obs, dict):
            obs = asdict(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def main(timesteps: int = 1_000) -> None:
    env = OPTCGEnv()
    agent = PPOAgent(env)
    agent.train(timesteps)


if __name__ == "__main__":  # pragma: no cover
    main()
