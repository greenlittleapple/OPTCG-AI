#!/usr/bin/env python
from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, fields
import time
from typing import Any, List, get_args, get_origin

import numpy as np
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
    attack_power: int
    attack_power_opponent: int

    def values(self):
        return {
            "can_attack": int(self.can_attack),
            "can_blocker": int(self.can_blocker),
            "can_choose_from_top": int(self.can_choose_from_top),
            "can_choose_friendly_target": int(self.can_choose_friendly_target),
            "can_choose_enemy_target": int(self.can_choose_enemy_target),
            "can_deploy": int(self.can_deploy),
            "can_draw": int(self.can_draw),
            "can_end_turn": int(self.can_end_turn),
            "can_resolve": int(self.can_resolve),
            "choice_cards": np.array(self.choice_cards),
            "hand": np.array(self.hand),
            "board": np.array(self.board),
            "board_opponent": np.array(self.board_opponent),
            "rested_cards": np.array(self.rested_cards),
            "rested_cards_opponent": np.array(self.rested_cards_opponent),
            "num_active_don": int(self.num_active_don),
            "num_active_don_opponent": int(self.num_active_don_opponent),
            "num_life": int(self.num_life),
            "num_life_opponent": int(self.num_life_opponent),
            "attack_power": int(self.attack_power),
            "attack_power_opponent": int(self.attack_power_opponent),
        }

class OPTCGEnv(AECEnv):
    """Minimal OPTCGSim environment following the PettingZoo AEC API."""

    metadata = {
        "name": "optcg_v1",
        "is_parallelizable": True,
    }
    render_mode = None
    fake_obs = None
    FAST_MODE = True
    VERBOSE = True

    @staticmethod
    def _build_obs_space() -> spaces.Dict:
        """Generate an observation space based on :class:`OPTCGPlayerObs`."""
        space_mapping: dict[str, spaces.Space] = {}
        for field in fields(OPTCGPlayerObs):
            typ = field.type
            name = field.name
            if typ == "bool":
                space_mapping[name] = spaces.Discrete(2)
            elif typ == "int":
                space_mapping[name] = spaces.Discrete(11)
            elif "List" in str(typ):
                space_mapping[name] = spaces.Box(shape=(5 if "board" in name or "choice_cards" in name or "rested_cards" in name else 10,), low=0, high=13000, dtype=np.int64)
        return spaces.Dict(space_mapping)

    def __init__(self, max_steps: int = 100, step_delay: float = 0.5) -> None:
        super().__init__()
        # Instantiate the vision helper used for observation gathering
        self._vision = finder.OPTCGVision()
        self._max_steps = max_steps
        self._delay = step_delay
        self._steps = 0

        self.possible_agents = ["player_0", "player_1"]
        self.agents: List[str] = []

        self.action_spaces = {agent: spaces.Discrete(25) for agent in self.possible_agents}
        obs_space = self._build_obs_space()
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

    def observe(self, agent: str) -> Any:
        def process_card_names(cards: List[str]) -> list:
            card_ids = []
            for card in cards:
                card_id = int(card[2:].replace('-', '')) if card != '' else 0
                card_ids.append(card_id)
            return card_ids

        if self.FAST_MODE and self.fake_obs:
            obs = copy(self.fake_obs)
        else:
            proceed = False
            while not proceed:
                obs = self._vision.scan()
                self.fake_obs = copy(obs)
                if obs.can_return_cards:
                    GUI.click_action0()
                    continue
                proceed = True

        obs.hand_p1 = process_card_names(obs.hand_p1)
        obs.hand_p2 = process_card_names(obs.hand_p2)
        obs.board_p1 = process_card_names(obs.board_p1)
        obs.board_p2 = process_card_names(obs.board_p2)
        obs.choice_cards = process_card_names(obs.choice_cards)

        agent_is_p1 = agent == "player_0"

        raw_power_self = obs.attack_powers[1] if agent_is_p1 else obs.attack_powers[0]
        raw_power_opp = obs.attack_powers[0] if agent_is_p1 else obs.attack_powers[1]

        def scale_power(val: int) -> int:
            return val if val == -1 else val // 1000

        attack_power = scale_power(raw_power_self)
        attack_power_opponent = scale_power(raw_power_opp)

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
            hand=obs.hand_p1 if agent_is_p1 else obs.hand_p2,
            board=obs.board_p1 if agent_is_p1 else obs.board_p2,
            board_opponent=obs.board_p2 if agent_is_p1 else obs.board_p1,
            rested_cards=obs.rested_cards_p1 if agent_is_p1 else obs.rested_cards_p2,
            rested_cards_opponent=obs.rested_cards_p2 if agent_is_p1 else obs.rested_cards_p1,
            num_active_don=obs.num_active_don_p1 if agent_is_p1 else obs.num_active_don_p2,
            num_active_don_opponent=obs.num_active_don_p2 if agent_is_p1 else obs.num_active_don_p1,
            num_life=obs.num_life_p1 if agent_is_p1 else obs.num_life_p2,
            num_life_opponent=obs.num_life_p2 if agent_is_p1 else obs.num_life_p1,
            attack_power=attack_power,
            attack_power_opponent=attack_power_opponent,
        ).values()

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
        self._last_obs = {agent: self.observe("player_0") for agent in self.agents}

    def switch_player(self) -> None:
        """Toggle :attr:`agent_selection` between ``player_0`` and ``player_1``."""
        self.agent_selection = (
            "player_1" if self.agent_selection == "player_0" else "player_0"
        )

    def action_mask(self, agent: str | None = None) -> list[int]:
        if agent is None:
            agent = self.agent_selection
        n = self.action_spaces[agent].__getattribute__("n")
        return [1] * n

    def step(self, action: int) -> None:
        """Execute *action* and update environment state."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        self._clear_rewards()
        reward = 0.0
        if action == 20:
            if self.agent_selection == "player_0":
                reward = 1
            else:
                reward = -1
        if action == 14:
            if self.agent_selection == "player_1":
                reward = 1
            else:
                reward = -1

        self.rewards[self.agent_selection] = reward
        self._last_obs["player_0"] = self.observe("player_0")
        self._last_obs["player_1"] = self.observe("player_1")

        self._steps += 1
        if self._steps % 10 == 0:
            self.switch_player()
        if self._steps >= self._max_steps:
            self.terminations["player_0"] = True
            self.terminations["player_1"] = True

        if self.VERBOSE:
            print(f"--- ACTION: {action}")
            print(f"--- OBS: {self._last_obs[self.agent_selection]}")

        self._accumulate_rewards()

    def last(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
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

    # Player 1 takes five actions
    for i in range(5):
        action = env.action_spaces[env.agent_selection].sample()
        env.step(action)
        obs, reward, terminated, truncated, _ = env.last()
        print(
            f"p1 step {i}: a={action} r={reward} term={terminated} trunc={truncated}"
        )
        if terminated or truncated:
            print("final obs ->", obs)
            return

    # Switch to Player 2
    env.switch_player()
    for i in range(4):
        action = env.action_spaces[env.agent_selection].sample()
        env.step(action)
        obs, reward, terminated, truncated, _ = env.last()
        print(
            f"p2 step {i}: a={action} r={reward} term={terminated} trunc={truncated}"
        )
        if terminated or truncated:
            print("final obs ->", obs)
            return

    # Switch back to Player 1 for a final action
    env.switch_player()
    action = env.action_spaces[env.agent_selection].sample()
    env.step(action)
    obs, reward, terminated, truncated, _ = env.last()
    print(
        f"p1 step 5: a={action} r={reward} term={terminated} trunc={truncated}"
    )
    print("final obs ->", obs)


if __name__ == "__main__":
    main()
