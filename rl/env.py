#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List

from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

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
    """Minimal OPTCGSim environment using the PettingZoo AEC API."""

    metadata = {"name": "optcg_aec"}

    _acting_player: int = 0

    def __init__(self, max_steps: int = 50, step_delay: float = 0.5) -> None:
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        self._action_spaces = {agent: spaces.Discrete(3) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "can_attack": spaces.Discrete(2),
                    "can_resolve": spaces.Discrete(2),
                    "can_end_turn": spaces.Discrete(2),
                }
            )
            for agent in self.possible_agents
        }

        self.action_spaces = self._action_spaces
        self.observation_spaces = self._observation_spaces

        self._vision = finder.OPTCGVision()
        self._max_steps = max_steps
        self._delay = step_delay
        self._steps = 0

        self._agent_selector = agent_selector(self.possible_agents)
        self.agents: List[str] = []
        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict[str, Any]] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self._observations: Dict[str, OPTCGPlayerObs] = {}

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
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed=seed)
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._steps = 0
        self._acting_player = self.agent_name_mapping[self.agent_selection]
        self._observations[self.agent_selection] = self.scan_and_process()

    def observe(self, agent: str) -> OPTCGPlayerObs | None:
        return self._observations.get(agent)

    def step(self, action: int | None, debug: bool = False) -> None:
        """Execute *action* for the current agent."""
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        reward = 0.0
        # action goes here
        time.sleep(self._delay)

        self._observations[agent] = self.scan_and_process()
        if self._observations[agent].can_resolve:
            reward += 1.0

        self.rewards[agent] = reward
        self._steps += 1
        if self._steps >= self._max_steps:
            for a in self.agents:
                self.truncations[a] = True

        if debug:
            print(f"--- ACTION: {action}")
            print(f"--- OBS: {self._observations[agent]}")

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()
        self._acting_player = self.agent_name_mapping[self.agent_selection]
        if self.agents:
            self._observations[self.agent_selection] = self.scan_and_process()


def main(num_steps: int = 10) -> None:
    """Run a short random rollout for quick manual testing."""
    env = OPTCGEnv(max_steps=num_steps)
    env.reset()
    for t, agent in enumerate(env.agent_iter(num_steps)):
        obs, reward, terminated, truncated, _ = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        action = env.action_space(agent).sample()
        env.step(action, debug=True)
        print(f"{agent} step {t}: a={action} r={reward} term={terminated} trunc={truncated}")


if __name__ == "__main__":
    main()
