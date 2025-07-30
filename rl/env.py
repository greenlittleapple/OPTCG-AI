#!/usr/bin/env python
from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, fields
from enum import Enum
import time
from typing import Any, List

import numpy as np
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

import gymnasium as gym
from utils import constants
from utils.gui import gui_macros
from utils.vision import finder
import functools

def timeit(func):
    """Decorator that prints execution time of the function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start

        print(f"Function {func.__name__}({signature}) took {elapsed:.4f} seconds")
        return result
    return wrapper


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

    def to_dict(self):
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
    last_obs = None
    FAST_MODE = False
    VERBOSE = False
    P1 = "player_0"
    P2 = "player_1"

    turn_state = {"total_don_p1": 3, "total_don_p2": 4, "has_attached_don": False}

    class INTENTS(Enum):
        END_TURN = "end_turn"
        ATTACH_DON = "attach_don"
        DEPLOY = "deploy"
        ATTACK = "attack"
        COUNTER = "counter"
        RESOLVE = "resolve"

    INTENTS_MAX = {
        INTENTS.END_TURN: 1,
        INTENTS.ATTACH_DON: 10,
        INTENTS.DEPLOY: 10,
        INTENTS.ATTACK: 6,
        INTENTS.COUNTER: 10,
        INTENTS.RESOLVE: 1,
    }

    INTENTS_ORDER = [
        INTENTS.END_TURN,
        INTENTS.ATTACH_DON,
        INTENTS.DEPLOY,
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

    def _get_hand_size(self, obs: dict) -> int:
        return np.count_nonzero(obs["hand"])

    def __init__(self, max_steps: int = 200, step_delay: float = 0.5) -> None:
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

    def observe(self, agent: str, new_obs = True) -> Any:
        def process_card_names(cards: List[str]) -> list[int]:
            card_ids = []
            for card in cards:
                card_id = constants.CARD_IDS.get(card, 0)
                card_ids.append(card_id)
            return card_ids

        if (not new_obs) and self.last_obs:
            obs = copy(self.last_obs)
        else:
            obs = self._vision.scan()
            self.last_obs = copy(obs)

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
            hand_size_opponent=(
                np.count_nonzero(hand_p2) if agent_is_p1 else np.count_nonzero(hand_p1)
            ),
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
        ).to_dict()
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
        gui_macros.reset_game()
        self._last_obs = {self.P1: self.observe(self.P1), self.P2: self.observe(self.P2, new_obs=False)}
        self.turn_state = {"total_don_p1": self._last_obs[self.P1]['num_active_don'], "total_don_p2": self._last_obs[self.P2]['num_active_don'], "has_attached_don": False}

    def switch_player(self) -> None:
        """Toggle :attr:`agent_selection` between ``player_0`` and ``player_1``."""
        self.agent_selection = (
            "player_1" if self.agent_selection == "player_0" else "player_0"
        )

    def create_action_mask(self, obs: dict[str, Any]) -> np.ndarray:
        is_countering = obs["is_countering"]
        num_don = int(obs.get("num_active_don", 0))
        leader_rested = bool(obs.get("leader_rested", 0))
        is_max_hand_size = self._get_hand_size(obs) >= 9

        # Handle End Turn logic
        end_turn_mask = [not is_countering and leader_rested and not is_max_hand_size]

        # Handle DON! Attach logic
        attach_mask = (
            [0] * 10
            if is_countering or is_max_hand_size or leader_rested or self.turn_state["has_attached_don"]
            else [1] * num_don + [0] * (10 - num_don)
        )

        # Handle Deploy logic
        deploy_mask = (
            [not is_countering and card != 0 for card in obs["hand"]]
            if is_max_hand_size
            else [0] * self.INTENTS_MAX[self.INTENTS.DEPLOY]
        )

        # Handle Attack logic
        if not (leader_rested or is_max_hand_size or is_countering):
            rested = list(obs["rested_cards_opponent"])
            attack_target_mask = [1] + rested
        else:
            attack_target_mask = [0] * (self.INTENTS_MAX[self.INTENTS.ATTACK])

        # Handle Counter / Resolve logic
        counter_mask = [is_countering and card != 0 for card in obs["hand"]]
        resolve_mask = [is_countering]

        mask: list = end_turn_mask
        mask.extend(attach_mask)
        mask.extend(deploy_mask)
        mask.extend(attack_target_mask)
        mask.extend(counter_mask)
        mask.extend(resolve_mask)
        return np.array(mask, dtype=np.int8)

    def end_game(self):
        self.terminations["player_0"] = True
        self.terminations["player_1"] = True

    def step(self, action: int) -> None:
        """Execute *action* and update environment state."""
        is_player1 = self.agent_selection == self.agents[0]
        player_index = 0 if is_player1 else 1
        last_obs = self._last_obs[self.agent_selection]

        if last_obs["is_game_over"]:
            self.end_game()

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return

        self._clear_rewards()

        # Action for the turn
        intent = None
        target = int(action)  # for easier math
        for temp_intent in self.INTENTS_ORDER:
            if target + 1 - self.INTENTS_MAX[temp_intent] <= 0:
                intent = temp_intent
                break
            target -= self.INTENTS_MAX[temp_intent]
        match intent:
            case self.INTENTS.END_TURN:
                gui_macros.end_turn()
            case self.INTENTS.ATTACH_DON:
                gui_macros.attach_don(
                    player_index,
                    0,
                    last_obs["num_active_don"],
                    self.turn_state["total_don_p1" if is_player1 else "total_don_p2"],
                    target + 1,
                )
            case self.INTENTS.DEPLOY:
                gui_macros.select_card(
                    player_index,
                    target,
                    self._get_hand_size(last_obs),
                    deploy_card=True,
                )
                if gui_macros._wait_for_button(constants.SELECT_CHARACTER_TO_REPLACE_BTN):
                    gui_macros._click_board_card(player_index, 1)
                while not gui_macros._wait_for_button(constants.END_TURN_BTN):
                    gui_macros.GUI.click_action0()
                    time.sleep(0.1)
            case self.INTENTS.ATTACK:
                gui_macros.attack(
                    player_index,
                    0,
                    int(not player_index),
                    0,
                )
            case self.INTENTS.COUNTER:
                gui_macros.select_card(
                    player_index,
                    target,
                    self._get_hand_size(last_obs),
                )
            case self.INTENTS.RESOLVE:
                assert gui_macros.click_action_when_visible(
                    0,
                    constants.RESOLVE_ATTACK_BTN,
                )
            case _:
                raise ValueError("Invalid intent!")
            
        # delay to let action happen
        # time.sleep(1)

        # Observe result
        self._last_obs["player_0"] = self.observe("player_0")
        self._last_obs["player_1"] = self.observe("player_1", new_obs=False)

        # Post-action processing
        match intent:
            case self.INTENTS.END_TURN:
                self.turn_state["total_don_p1"] = self._last_obs["player_0"][
                    "num_active_don"
                ]
                self.turn_state["total_don_p2"] = self._last_obs["player_1"][
                    "num_active_don"
                ]
                self.turn_state["has_attached_don"] = False
                self.switch_player()
            case self.INTENTS.ATTACH_DON:
                self.turn_state["total_don_p1" if is_player1 else "total_don_p2"] -= (target + 1)
                self.turn_state["has_attached_don"] = True
            case self.INTENTS.ATTACK:
                self.switch_player()
            case self.INTENTS.RESOLVE:
                if self._last_obs[self.agent_selection]['is_game_over']:
                    self.rewards[self.agents[player_index]] = -1
                    self.rewards[self.agents[int(not player_index)]] = 1
                    print(f"--- PLAYER {int(not player_index) + 1} WINS")
                self.switch_player()

        self._steps += 1
        if self._steps >= self._max_steps:
            self.end_game()

        if self.VERBOSE:
            print(f"--- ACTION: {action}")
            print(f"--- INTENT: {intent}; TARGET: {target}")
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

    def observe(self, agent: str, new_obs = True):
        obs = super().observe(agent, new_obs)
        if isinstance(obs, OPTCGPlayerObs):
            return asdict(obs)
        return obs

    def action_mask(self):
        return super().observe(self.agent_selection)["action_mask"]


def main(num_steps: int = 30) -> None:
    """Run a short random rollout for quick manual testing."""
    env = OPTCGEnv(max_steps=num_steps)
    env.reset()

    obs = env.observe(env.P1)
    print("reset ->", obs)

    for i in range(num_steps):
        print(f'--- LAST OBS: {env.last()}')
        action = env.action_spaces[env.agent_selection].sample(env.action_mask())
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"--- STEP: player={env.agent_selection} step={i}: a={action} r={reward} term={terminated} trunc={truncated}")
        if terminated or truncated:
            print("final obs ->", obs)
            return

if __name__ == "__main__":
    main()
