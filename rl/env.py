#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass, fields
import time
from typing import Any, List, get_args, get_origin

from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

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

class OPTCGEnv(AECEnv):
    """Minimal OPTCGSim environment following the PettingZoo AEC API."""

    metadata = {"name": "optcg_v1"}

    @staticmethod
    def _build_obs_space() -> spaces.Dict:
        """Generate an observation space based on :class:`OPTCGPlayerObs`."""
        space_mapping: dict[str, spaces.Space] = {}
        for field in fields(OPTCGPlayerObs):
            typ = field.type
            name = field.name
            if typ is bool:
                space_mapping[name] = spaces.Discrete(2)
            elif typ is int:
                space_mapping[name] = spaces.Discrete(11)
            elif get_origin(typ) in (list, List):
                subtype = get_args(typ)[0]
                if subtype is str:
                    space_mapping[name] = spaces.Sequence(spaces.Text(max_length=10))
                elif subtype is int:
                    space_mapping[name] = spaces.Sequence(spaces.Discrete(99))
        return spaces.Dict(space_mapping)

    def __init__(self, max_steps: int = 50, step_delay: float = 0.5) -> None:
        super().__init__()
        # Instantiate the vision helper used for observation gathering
        self._vision = finder.OPTCGVision()
        self._max_steps = max_steps
        self._delay = step_delay
        self._steps = 0

        self.possible_agents = ["player_0", "player_1"]
        self.agents: List[str] = []

        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.possible_agents}
        obs_space = self._build_obs_space()
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

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
            hand=obs.hand_p1,
            board=obs.board_p1,
            board_opponent=obs.board_p2,
            rested_cards=obs.rested_cards_p1,
            rested_cards_opponent=obs.rested_cards_p2,
            num_active_don=obs.num_active_don_p1,
            num_active_don_opponent=obs.num_active_don_p2,
            num_life=obs.num_life_p1,
            num_life_opponent=obs.num_life_p2,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._steps = 0
        self.agent_selection = self.agents[0]
        self._last_obs = {self.agents[0]: self.scan_and_process()}

    def step(self, action: int, debug: bool = False) -> None:
        """Execute *action* and update environment state."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        self._clear_rewards()

        # action goes here
        time.sleep(self._delay)

        obs = self.scan_and_process()
        reward = 1.0 if obs.can_resolve else 0.0
        self.rewards[self.agent_selection] = reward
        self._last_obs[self.agent_selection] = obs

        self._steps += 1
        if self._steps >= self._max_steps:
            self.truncations[self.agent_selection] = True

        if debug:
            print(f"--- ACTION: {action}")
            print(f"--- OBS: {obs}")

        self._accumulate_rewards()

    def last(self) -> tuple[OPTCGPlayerObs, float, bool, bool, dict[str, Any]]:
        """Return observation and info for the current agent."""
        agent = self.agent_selection
        return (
            self._last_obs[agent],
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )


def main(num_steps: int = 10) -> None:
    """Run a short random rollout for quick manual testing."""
    env = OPTCGEnv(max_steps=num_steps)
    env.reset()
    obs, _, terminated, truncated, _ = env.last()
    print("reset ->", obs)
    for t in range(num_steps):
        action = env.action_spaces[env.agent_selection].sample()
        env.step(action)
        obs, reward, terminated, truncated, _ = env.last()
        print(f"step {t}: a={action} r={reward} term={terminated} trunc={truncated}")
        if terminated or truncated:
            break
    print("final obs ->", obs)


if __name__ == "__main__":
    main()
