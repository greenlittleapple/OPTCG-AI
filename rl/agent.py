from __future__ import annotations

from dataclasses import asdict
import os
import re
from typing import Any

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env import OPTCGPlayerObs, OPTCGEnv


class PPOAgent:
    """Minimal PPO agent using Stable-Baselines3."""

    def __init__(self, env: OPTCGEnv, *, verbose: int = 2) -> None:
        env.reset()
        env = ActionMasker(env, lambda e: e.action_mask()) # type: ignore
        # check_env(env)

        self.model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=verbose,
            tensorboard_log="./logs/",
            n_steps=256,
            batch_size=32,
        )

    def train(self, timesteps: int = 1_000) -> None:
        """Train the agent for ``timesteps`` steps."""

        def load_latest_checkpoint_or_skip(model: MaskablePPO, folder, prefix="ppo_checkpoint_"):
            """
            If checkpoint files like `{prefix}{steps}_steps.zip` exist in folder,
            load the one with highest step count into the given model instance.
            Otherwise, do nothing.
            """
            # list matching .zip files
            files = [f for f in os.listdir(folder)
                if f.startswith(prefix) and f.endswith(".zip")]
            if not files:
                print(f"No checkpoints found in {folder}")
                return model
            def extract_steps(filename):
                m = re.search(rf"{re.escape(prefix)}(\d+)_steps\.zip", filename)
                return int(m.group(1)) if m else -1
            latest_file = max(files, key=extract_steps)
            latest_path = os.path.join(folder, latest_file)
            step_count = extract_steps(latest_file)
            print(f"Loading checkpoint from {latest_path} with {step_count} steps")
            model = model.load(latest_path, env=model.get_env(), device=model.device)
            model.num_timesteps = step_count  # ✅ manually restore step count
            return model

        checkpoint_cb = CheckpointCallback(
            save_freq=50, save_path="checkpoints/",
            name_prefix="ppo_checkpoint",
            save_vecnormalize=True,   # if using VecNormalize
        )

        self.model = load_latest_checkpoint_or_skip(self.model, './checkpoints')
        self.model.learn(total_timesteps=timesteps, progress_bar=True, callback=checkpoint_cb, reset_num_timesteps=False)

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
