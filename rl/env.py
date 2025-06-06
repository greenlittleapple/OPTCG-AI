#!/usr/bin/env python
from __future__ import annotations

import time
from typing import Any, Tuple

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - gym is optional for linting/test envs
    gym = None
    spaces = None

from utils.gui import gui_macros as MACROS
from utils.gui import gui_automation_starter as GUI
from utils.vision import finder


class OPTCGEnv(gym.Env if gym is not None else object):
    """Minimal OPTCGSim environment following the Gym API."""

    def __init__(self, max_steps: int = 50, step_delay: float = 0.5) -> None:
        super().__init__()
        # Instantiate the vision helper used for observation gathering
        self._vision = finder.OPTCGVision()
        self._max_steps = max_steps
        self._delay = step_delay
        self._steps = 0

        if spaces is not None:
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Dict(
                {
                    "can_attack": spaces.Discrete(2),
                    "can_resolve": spaces.Discrete(2),
                    "can_end_turn": spaces.Discrete(2),
                }
            )
        else:  # pragma: no cover
            self.action_space = None  # type: ignore
            self.observation_space = None  # type: ignore

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[dict[str, Any], dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)
        self._steps = 0
        obs = self._vision.scan()
        return obs, {}

    def step(self, action: int) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute *action* and return the new observation."""
        terminated = False
        truncated = False
        reward = 0.0

        if action == 0:  # Leader attacks leader
            MACROS.attack(1, 0, 2, 0)
        elif action == 1:  # End turn
            GUI.click_end_turn()
        time.sleep(self._delay)

        obs = self._vision.scan()
        if obs.get("can_resolve"):
            reward += 1.0
        self._steps += 1
        if self._steps >= self._max_steps:
            truncated = True
        return obs, reward, terminated, truncated, {}


def main(num_steps: int = 10) -> None:
    """Run a short random rollout for quick manual testing."""
    env = OPTCGEnv(max_steps=num_steps)
    obs, _ = env.reset()
    print("reset ->", obs)
    for t in range(num_steps):
        if gym is not None and env.action_space is not None:
            action = env.action_space.sample()
        else:
            action = t % 3  # deterministic fallback
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"step {t}: a={action} r={reward} term={terminated} trunc={truncated}")
        if terminated or truncated:
            break
    print("final obs ->", obs)


if __name__ == "__main__":
    main()
