"""
High-level GUI macros for OPTCGSim automation.
============================================

This module converts symbolic instructions ("P1 card 2 uses Action 3 on
P2 card 4") into concrete click sequences using helpers from
`utils.gui.gui_automation_starter`.

Public API
----------
* perform_action() – generic helper covering no‑target, single‑target, and
  multi‑target card actions.
* attack()         – convenience wrapper for the common "Action 1 = Attack"
  pattern that lets you specify any acting *and* target card.
* deploy_card()    – select a card from hand and confirm with Action 1.
* attach_don()     – attach one or more DON!! cards to a specified card on
  the active player's board.

All text is ASCII‑only; no smart quotes.
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
        Each tuple is (player, card_index). Provide None (or empty list) for
        no‑target actions.
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
    """Attack macro (Action 1).

    Examples
    --------
    >>> attack(1, 0, 2, 0)  # P1 leader attacks P2 leader
    >>> attack(2, 3, 1, 2)  # P2 slot‑3 card attacks P1 slot‑2 card
    """
    perform_action(
        acting_player=acting_player,
        acting_card_index=acting_card_index,
        action_number=1,
        targets=[(target_player, target_card_index)],
    )


def deploy_card(
    acting_player: int,
    hand_card_index: int,
    hand_size: int,
) -> None:
    """Deploy a card from the acting player's hand and confirm.

    Parameters
    ----------
    acting_player : int
        1 for Player 1, 2 for Player 2.
    hand_card_index : int
        Index of the card to deploy (0 = left‑most).
    hand_size : int
        Total number of cards currently in that hand.
    """
    # --- Validate ----------------------------------------------------
    if acting_player not in (1, 2):
        raise ValueError("acting_player must be 1 or 2")
    if hand_size < 1:
        raise ValueError("hand_size must be at least 1")
    if not 0 <= hand_card_index < hand_size:
        raise ValueError("hand_card_index must be within hand_size")

    # --- Select card from hand --------------------------------------
    _click_hand_card(acting_player, hand_card_index, hand_size)

    # --- Confirm with Action 1 --------------------------------------
    GUI.click_action1()


def attach_don(
    acting_player: int,
    card_index: int,
    attachable_don: int,
    total_don: int,
    num_to_attach: int,
) -> None:
    """Attach DON!! cards to a card on the active player's board.

    Parameters
    ----------
    acting_player : int
        1 or 2 (whose turn it is). Only that player can attach DON.
    card_index : int
        0 = leader, 1–5 = board slots.
    attachable_don : int
        Number of DON currently in the "attachable" state (always the last
        cards in the DON row).
    total_don : int
        Total DON the player controls (0–10).
    num_to_attach : int
        How many DON to attach to the chosen card.
    """
    # --- Validate ----------------------------------------------------
    if acting_player not in (1, 2):
        raise ValueError("acting_player must be 1 or 2")
    if not 0 <= card_index <= 5:
        raise ValueError("card_index must be 0–5")
    if not 0 <= attachable_don <= total_don <= 10:
        raise ValueError("attachable_don ≤ total_don ≤ 10 must hold")
    if not 1 <= num_to_attach <= attachable_don:
        raise ValueError("num_to_attach must be 1..attachable_don")

    # --- Determine which DON helper to use --------------------------
    click_don = GUI.click_p1_don if acting_player == 1 else GUI.click_p2_don

    # --- Compute DON indices to click -------------------------------
    first_to_click = total_don - num_to_attach
    don_indices = range(first_to_click, total_don)  # inclusive‑exclusive

    # --- Click chosen DON ------------------------------------------
    for d_idx in don_indices:
        click_don(d_idx)

    # --- Click card to receive DON ---------------------------------
    _click_board_card(acting_player, card_index)

    # --- Confirm attachment (Action 1) ------------------------------
    GUI.click_action1()

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


def _click_hand_card(player: int, idx: int, hand_size: int) -> None:
    if player == 1:
        GUI.click_p1_hand(card_index=idx, hand_size=hand_size)
    else:
        GUI.click_p2_hand(card_index=idx, hand_size=hand_size)
