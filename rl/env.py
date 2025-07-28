#!/usr/bin/env python
from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, List

import numpy as np
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

import gymnasium as gym
from utils import constants
from utils.gui import gui_macros
from utils.vision import finder


@dataclass
class OPTCGPlayerObs:
    """Observation data returned to the learning agent."""

    choice_cards: List[int]
    hand: List[int]
    hand_size_opponent: int
    board: List[int]
    board_opponent: List[int]
    rested_cards: List[int]
    rested_cards_opponent: List[int]
    leader_rested: bool
    leader_rested_opponent: bool
    num_active_don: int
    num_active_don_opponent: int
    num_life: int
    num_life_opponent: int
    attack_power: int
    attack_power_opponent: int
    is_countering: bool
    is_game_over: bool

    def values(self):
        return {
            "choice_cards": np.array(self.choice_cards),
            "hand": np.array(self.hand),
            "hand_size_opponent": int(self.hand_size_opponent),
            "board": np.array(self.board),
            "board_opponent": np.array(self.board_opponent),
            "rested_cards": np.array(self.rested_cards),
            "rested_cards_opponent": np.array(self.rested_cards_opponent),
            "leader_rested": int(self.leader_rested),
            "leader_rested_opponent": int(self.leader_rested_opponent),
            "num_active_don": int(self.num_active_don),
            "num_active_don_opponent": int(self.num_active_don_opponent),
            "num_life": int(self.num_life),
            "num_life_opponent": int(self.num_life_opponent),
            "attack_power": int(self.attack_power),
            "attack_power_opponent": int(self.attack_power_opponent),
            "is_countering": int(self.is_countering),
            "is_game_over": int(self.is_game_over),
        }


class OPTCGEnvBase(AECEnv):
    """Minimal OPTCGSim environment following the PettingZoo AEC API."""

    metadata = {
        "name": "optcg_v1",
        "is_parallelizable": True,
    }
    render_mode = None
    fake_obs = None
    FAST_MODE = True
    VERBOSE = True
    P1 = "player_0"
    P2 = "player_1"

    class INTENTS(Enum):
        END_TURN = "end_turn"
        ATTACH_DON = "attach_don"
        ATTACK = "attack"
        COUNTER = "counter"
        RESOLVE = "resolve"

    INTENTS_MAX = {
        INTENTS.END_TURN: 1,
        INTENTS.ATTACH_DON: 10,
        INTENTS.ATTACK: 6,
        INTENTS.COUNTER: 10,
        INTENTS.RESOLVE: 1,
    }

    INTENTS_ORDER = [
        INTENTS.END_TURN,
        INTENTS.ATTACH_DON,
        INTENTS.ATTACK,
        INTENTS.COUNTER,
        INTENTS.RESOLVE,
    ]

    TOTAL_ACTIONS = sum(INTENTS_MAX.values())

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
                space_mapping[name] = spaces.Discrete(100)
            elif "List" in str(typ):
                space_mapping[name] = spaces.Box(
                    shape=(
                        (
                            5
                            if "board" in name
                            or "choice_cards" in name
                            or "rested_cards" in name
                            else 10
                        ),
                    ),
                    low=0,
                    high=16,
                    dtype=np.int64,
                )
        space_mapping["action_mask"] = spaces.MultiBinary(OPTCGEnvBase.TOTAL_ACTIONS)
        return spaces.Dict(space_mapping)

    @staticmethod
    def _build_action_space() -> spaces.Discrete:
        return spaces.Discrete(OPTCGEnvBase.TOTAL_ACTIONS)

    def __init__(self, max_steps: int = 100, step_delay: float = 0.5) -> None:
        super().__init__()
        # Instantiate the vision helper used for observation gathering
        self._vision = finder.OPTCGVision()
        self._max_steps = max_steps
        self._delay = step_delay
        self._steps = 0

        self.possible_agents = ["player_0", "player_1"]
        self.agents: List[str] = []

        self.action_spaces = {
            agent: self._build_action_space() for agent in self.possible_agents
        }
        obs_space = self._build_obs_space()
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

    def observe(self, agent: str) -> Any:
        def process_card_names(cards: List[str]) -> list[int]:
            card_ids = []
            for card in cards:
                card_id = constants.CARD_IDS.get(card, 0)
                card_ids.append(card_id)
            return card_ids

        if self.FAST_MODE and self.fake_obs:
            obs = copy(self.fake_obs)
        else:
            gui_macros.click_action_when_visible(0, constants.RETURN_CARDS_TO_DECK_BTN)
            obs = self._vision.scan()
            self.fake_obs = copy(obs)

        hand_p1 = process_card_names(obs.hand_p1)
        hand_p2 = process_card_names(obs.hand_p2)
        board_p1 = process_card_names(obs.board_p1)
        board_p2 = process_card_names(obs.board_p2)
        choice_cards = process_card_names(obs.choice_cards)

        agent_is_p1 = agent == "player_0"

        raw_power_self = obs.attack_powers[1] if agent_is_p1 else obs.attack_powers[0]
        raw_power_opp = obs.attack_powers[0] if agent_is_p1 else obs.attack_powers[1]

        def scale_power(val: int) -> int:
            if val == -1:
                return 99
            return val // 1000

        attack_power = scale_power(raw_power_self)
        attack_power_opponent = scale_power(raw_power_opp)

        obs_dict = OPTCGPlayerObs(
            choice_cards=choice_cards,
            hand=hand_p1 if agent_is_p1 else hand_p2,
            hand_size_opponent=np.count_nonzero(hand_p2) if agent_is_p1 else np.count_nonzero(hand_p1),
            board=board_p1 if agent_is_p1 else board_p2,
            board_opponent=board_p2 if agent_is_p1 else board_p1,
            rested_cards=obs.rested_cards_p1 if agent_is_p1 else obs.rested_cards_p2,
            rested_cards_opponent=(
                obs.rested_cards_p2 if agent_is_p1 else obs.rested_cards_p1
            ),
            leader_rested=(
                obs.leader_rested_p1 if agent_is_p1 else obs.leader_rested_p2
            ),
            leader_rested_opponent=(
                obs.leader_rested_p2 if agent_is_p1 else obs.leader_rested_p1
            ),
            num_active_don=(
                obs.num_active_don_p1 if agent_is_p1 else obs.num_active_don_p2
            ),
            num_active_don_opponent=(
                obs.num_active_don_p2 if agent_is_p1 else obs.num_active_don_p1
            ),
            num_life=obs.num_life_p1 if agent_is_p1 else obs.num_life_p2,
            num_life_opponent=obs.num_life_p2 if agent_is_p1 else obs.num_life_p1,
            attack_power=attack_power,
            attack_power_opponent=attack_power_opponent,
            is_countering=obs.is_countering,
            is_game_over=obs.is_game_over,
        ).values()
        obs_dict["action_mask"] = self.create_action_mask(obs_dict)
        return obs_dict

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> None:
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

    def create_action_mask(self, obs: dict[str, Any]) -> np.ndarray:
        is_countering = obs["is_countering"]

        # Handle End Turn logic
        end_turn_mask = [not is_countering]

        # Handle DON! Attach logic
        num_don = int(obs.get("num_active_don", 0))
        attach_mask = [0] * 10 if is_countering else [1] * num_don + [0] * (10 - num_don)

        # Handle Attack logic
        leader_rested = bool(obs.get("leader_rested", 0))
        if not leader_rested:
            rested = list(obs["rested_cards_opponent"])
            attack_target_mask = [1] + rested
        else:
            attack_target_mask = [0] * (self.INTENTS_MAX[self.INTENTS.ATTACK])

        # Handle Counter / Resolve logic
        counter_mask = [is_countering and card != 0 for card in obs["hand"]]
        resolve_mask = [is_countering]

        mask: list = end_turn_mask
        mask.extend(attach_mask)
        mask.extend(attack_target_mask)
        mask.extend(counter_mask)
        mask.extend(resolve_mask)
        return np.array(mask, dtype=np.int8)

    def step(self, action: int) -> None:
        """Execute *action* and update environment state."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        self._clear_rewards()
        reward = 0.0

        intent = None
        target = int(action)  # for easier math
        for temp_intent in self.INTENTS_ORDER:
            if target + 1 - self.INTENTS_MAX[temp_intent] <= 0:
                intent = temp_intent
                break
            target -= self.INTENTS_MAX[temp_intent]

        match intent:
            case self.INTENTS.END_TURN:
                pass
            case self.INTENTS.ATTACH_DON:
                pass
            case self.INTENTS.ATTACK:
                pass
            case self.INTENTS.COUNTER:
                pass
            case self.INTENTS.RESOLVE:
                pass
            case _:
                raise ValueError('Invalid intent!')


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
            print(f'--- INTENT: {intent}; TARGET: {target}')
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


class OPTCGEnv(OPTCGEnvBase, gym.Env):
    """Wrapper to adapt PettingZoo AEC environments for SB3 with action masking."""

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self.observation_space = super().observation_space(self.possible_agents[0])  # type: ignore
        self.action_space = super().action_space(self.possible_agents[0])  # type: ignore
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
        return super().create_action_mask(self._last_obs[self.agent_selection])


def main(num_steps: int = 10) -> None:
    """Run a short random rollout for quick manual testing."""
    env = OPTCGEnv(max_steps=num_steps)
    env.reset()

    obs = env.observe(env.P1)
    print("reset ->", obs)

    # Player 1 takes five actions
    for i in range(5):
        action = env.action_spaces[env.agent_selection].sample(env.action_mask())
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"p1 step {i}: a={action} r={reward} term={terminated} trunc={truncated}")
        if terminated or truncated:
            print("final obs ->", obs)
            return

    # Switch to Player 2
    env.switch_player()
    for i in range(4):
        action = env.action_spaces[env.agent_selection].sample(env.action_mask())
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"p2 step {i}: a={action} r={reward} term={terminated} trunc={truncated}")
        if terminated or truncated:
            print("final obs ->", obs)
            return

    # Switch back to Player 1 for a final action
    env.switch_player()
    action = env.action_spaces[env.agent_selection].sample(env.action_mask())
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"p1 step 5: a={action} r={reward} term={terminated} trunc={truncated}")
    print("final obs ->", obs)


if __name__ == "__main__":
    main()
