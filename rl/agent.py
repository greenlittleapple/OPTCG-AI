from __future__ import annotations

from dataclasses import asdict
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env import OPTCGPlayerObs, OPTCGEnv


class PPOAgent:
    """Minimal PPO agent using Stable-Baselines3."""

    def __init__(self, env: OPTCGEnv, *, verbose: int = 2) -> None:
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
