"""
High-level GUI macros for OPTCGSim automation.
============================================

This module turns symbolic game instructions ("P1 card 2 uses Action 3 on
P2 card 4") into the concrete click sequences required by the GUI helpers
in `utils.gui.gui_automation_starter`.

Public API
----------
* perform_action() – generic helper covering no‑target, single‑target, and
  multi‑target card actions.
* attack()         – thin convenience wrapper for the common "Action 1 =
  Attack" pattern that now lets you specify any acting *and* target card.

--------------------------------------------------------------------------
ASCII‑only; no smart quotes.
"""
from __future__ import annotations

from typing import List, Tuple

from utils.gui import gui_automation_starter as GUI

# ---------------------------------------------------------------------
# Generic helper -------------------------------------------------------
# ---------------------------------------------------------------------

def perform_action(
    acting_player: int,
    acting_card_index: int,
    action_number: int,
    targets: List[Tuple[int, int]] | None = None,
) -> None:
    """Execute an in‑game action.

    Parameters
    ----------
    acting_player : int
        1 for Player 1, 2 for Player 2.
    acting_card_index : int
        0 selects the leader; 1–5 select board card slots.
    action_number : int
        0, 1, 2, or 3 → mapped to Action‑button helpers.
    targets : list[tuple[int, int]] | None
        Each tuple is (player, card_index). Provide an empty list or None
        for no‑target actions.
    """
    # --- Validate ----------------------------------------------------
    if acting_player not in (1, 2):
        raise ValueError("acting_player must be 1 or 2")
    if not 0 <= acting_card_index <= 5:
        raise ValueError("acting_card_index must be 0–5")
    if action_number not in (0, 1, 2, 3):
        raise ValueError("action_number must be 0–3")
    if targets is None:
        targets = []
    for t_player, t_idx in targets:
        if t_player not in (1, 2):
            raise ValueError("target player must be 1 or 2")
        if not 0 <= t_idx <= 5:
            raise ValueError("target card index must be 0–5")

    # --- Select acting card -----------------------------------------
    _click_board_card(acting_player, acting_card_index)

    # --- Click action button ----------------------------------------
    _click_action_button(action_number)

    # --- Click each target -----------------------------------------
    for t_player, t_idx in targets:
        _click_board_card(t_player, t_idx)

# ---------------------------------------------------------------------
# Convenience wrappers -------------------------------------------------
# ---------------------------------------------------------------------

def attack(
    acting_player: int,
    acting_card_index: int,
    target_player: int,
    target_card_index: int,
) -> None:
    """Common attack macro (Action 1).

    Lets you specify any card as the attacker and any card as the target.

    Example
    -------
    >>> attack(1, 0, 2, 0)  # P1 leader attacks P2 leader
    >>> attack(2, 3, 1, 2)  # P2 slot‑3 card attacks P1 slot‑2 card
    """
    perform_action(
        acting_player=acting_player,
        acting_card_index=acting_card_index,
        action_number=1,
        targets=[(target_player, target_card_index)],
    )

# ---------------------------------------------------------------------
# Internal util --------------------------------------------------------
# ---------------------------------------------------------------------

_ACTION_MAP = {
    0: GUI.click_action0,
    1: GUI.click_action1,
    2: GUI.click_action2,
    3: GUI.click_action3,
}


def _click_action_button(num: int) -> None:
    _ACTION_MAP[num]()


def _click_board_card(player: int, idx: int) -> None:
    if player == 1:
        if idx == 0:
            GUI.click_p1_leader()
        else:
            GUI.click_p1_card(idx - 1)
    else:  # player 2
        if idx == 0:
            GUI.click_p2_leader()
        else:
            GUI.click_p2_card(idx - 1)
