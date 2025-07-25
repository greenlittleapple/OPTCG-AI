#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, List, Tuple

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - gym is optional for linting/test envs
    gym = None
    spaces = None

from utils.gui import gui_macros as MACROS
from utils.gui import gui_automation_starter as GUI
from utils.vision import finder

@dataclass
class OPTCGPlayerObs:
    can_attack: bool
    can_blocker: bool
    can_choose_from_top: bool
    can_choose_friendly_target: bool
    can_choose_enemy_target: bool
    can_deploy: bool
    can_draw: bool
    can_end_turn: bool
    can_resolve: bool
    choice_cards: List[str]
    hand: List[str]
    board: List[str]
    board_opponent: List[str]
    rested_cards: List[int]
    rested_cards_opponent: List[int]
    num_active_don: int
    num_active_don_opponent: int
    num_life: int
    num_life_opponent: int

class OPTCGEnv(gym.Env if gym is not None else object):
    """Minimal OPTCGSim environment following the Gym API."""

    _acting_player: int = 0

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

    def scan_and_process(self) -> OPTCGPlayerObs:
        proceed = False
        while not proceed:
            obs = self._vision.scan()
            if obs.can_return_cards:
                GUI.click_action0()
                continue
            proceed = True
        return OPTCGPlayerObs(
            can_attack=obs.can_attack,
            can_blocker=obs.can_blocker,
            can_choose_from_top=obs.can_choose_from_top,
            can_choose_friendly_target=obs.can_choose_friendly_target,
            can_choose_enemy_target=obs.can_choose_enemy_target,
            can_deploy=obs.can_deploy,
            can_draw=obs.can_draw,
            can_end_turn=obs.can_end_turn,
            can_resolve=obs.can_resolve,
            choice_cards=obs.choice_cards,
            hand=obs.hand_p1 if self._acting_player == 0 else obs.hand_p2,
            board=obs.board_p1 if self._acting_player == 0 else obs.board_p2,
            board_opponent=obs.board_p2 if self._acting_player == 0 else obs.board_p1,
            rested_cards=obs.rested_cards_p1 if self._acting_player == 0 else obs.rested_cards_p2,
            rested_cards_opponent=obs.rested_cards_p2 if self._acting_player == 0 else obs.rested_cards_p1,
            num_active_don=obs.num_active_don_p1 if self._acting_player == 0 else obs.num_active_don_p2,
            num_active_don_opponent=obs.num_active_don_p2 if self._acting_player == 0 else obs.num_active_don_p1,
            num_life=obs.num_life_p1 if self._acting_player == 0 else obs.num_life_p2,
            num_life_opponent=obs.num_life_p2 if self._acting_player == 0 else obs.num_life_p1,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[OPTCGPlayerObs, dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)
        self._steps = 0
        obs = self.scan_and_process()
        return obs, {}

    def step(self, action: int, debug = False) -> Tuple[OPTCGPlayerObs, float, bool, bool, dict[str, Any]]:
        """Execute *action* and return the new observation."""
        terminated = False
        truncated = False
        reward = 0.0

        # action goes here
        time.sleep(self._delay)

        obs = self.scan_and_process()
        if obs.can_resolve:
            reward += 1.0
        self._steps += 1
        if self._steps >= self._max_steps:
            truncated = True
        if debug:
            print(f"--- ACTION: {action}")
            print(f"--- OBS: {obs}")
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
        obs, reward, terminated, truncated, _ = env.step(action, debug=True)
        print(f"step {t}: a={action} r={reward} term={terminated} trunc={truncated}")
        if terminated or truncated:
            break
    print("final obs ->", obs)


if __name__ == "__main__":
    main()
