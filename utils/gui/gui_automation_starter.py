#!/usr/bin/env python3
"""
OPTCGSim GUI‑Automation Starter Script
=====================================

All clicks are made relative to the OPTCGSim window, not the full screen.

Helpers included so far:
    • Menu buttons: Solo v Self, Start
    • In‑game buttons: Action 0–3, End Turn (double‑click Action 0)
    • Leaders: P1, P2
    • Board cards: P1 & P2 board slots with updated spacing
    • Hand cards: P1 and P2 variable‑width hand slots

*Board‑card spacing update*
---------------------------
P1 board slots are now spaced **6 %** apart (was 5 %).
First slot still starts at x = 40 %.

P2 board slots mirror this, but the entire row shifts **10 % leftward**
so the right‑most slot now starts at x = 60 %.

Dependencies:
    pyautogui
    pygetwindow

Adjust the WINDOW_TITLE constant if the client window has a different title.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import pyautogui as pag
import pygetwindow as gw  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_TITLE = "OPTCGSim"  # Substring that identifies the game window
CLICK_DELAY = 0.2           # Seconds to wait after each click

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_optcg_window() -> gw.Window:
    """Return the first window whose title contains WINDOW_TITLE (case-insensitive)."""
    titles = [t for t in gw.getAllTitles() if WINDOW_TITLE.lower() in t.lower()]
    if not titles:
        sys.exit(f"No window found with title containing '{WINDOW_TITLE}'.")

    win = gw.getWindowsWithTitle(titles[0])[0]
    if not win.isActive:
        win.activate()
        time.sleep(0.2)
    return win


def click_relative_to_window(rel_x: float, rel_y: float, delay: float = CLICK_DELAY) -> None:
    """Click at a position expressed as a fraction of the window size.

    Args:
        rel_x: Horizontal position (0.0–1.0) of the window width.
        rel_y: Vertical position (0.0–1.0) of the window height.
        delay: Seconds to pause after the click.
    """
    if not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
        raise ValueError("rel_x and rel_y must each be between 0.0 and 1.0")

    win = _get_optcg_window()
    x = int(win.left + rel_x * win.width)
    y = int(win.top + rel_y * win.height)
    pag.click(x, y)
    time.sleep(delay)

# ---------------------------------------------------------------------------
# High‑level button helpers
# ---------------------------------------------------------------------------

# Menu buttons

def click_solo_vs_self() -> None:
    """Click the Solo v Self button (center, 60% down)."""
    click_relative_to_window(0.5, 0.6)


def click_start() -> None:
    """Click the Start button (20% from left, 60% down)."""
    click_relative_to_window(0.2, 0.6)

# In‑game controls

def click_action0() -> None:
    """Click Action 0 (contextual button at 80% from left, 90% down)."""
    click_relative_to_window(0.8, 0.9)


def click_end_turn() -> None:
    """Click End Turn with confirmation by clicking Action 0 twice."""
    click_action0()
    click_action0()


def click_p1_leader() -> None:
    """Click Player 1 leader (55% from left, 75% down)."""
    click_relative_to_window(0.55, 0.75)


def click_p2_leader() -> None:
    """Click Player 2 leader (45% from left, 25% down)."""
    click_relative_to_window(0.45, 0.25)


def click_action1() -> None:
    """Click Action 1 (80% from left, 80% down)."""
    click_relative_to_window(0.8, 0.8)


def click_action2() -> None:
    """Click Action 2 (80% from left, 70% down)."""
    click_relative_to_window(0.8, 0.7)


def click_action3() -> None:
    """Click Action 3 (80% from left, 60% down)."""
    click_relative_to_window(0.8, 0.6)

# Board card helpers (updated spacing)

_P1_BOARD_START_X = 0.40   # first slot
_P1_BOARD_STEP_X  = 0.06   # 6% horizontal spacing (20% increase from 5%)
_P1_BOARD_Y       = 0.55

_P2_BOARD_START_X = 0.60   # shifted 10% left (was 0.70)
_P2_BOARD_STEP_X  = 0.06   # same 6% spacing, mirrored
_P2_BOARD_Y       = 0.45


def click_p1_card(slot: int) -> None:
    """Click Player 1's board card in the specified slot (0‑4, left→right).

    Spacing between slots is now 6 % of window width.
    """
    if not 0 <= slot <= 4:
        raise ValueError("slot must be between 0 and 4 inclusive")
    rel_x = _P1_BOARD_START_X + _P1_BOARD_STEP_X * slot
    click_relative_to_window(rel_x, _P1_BOARD_Y)


def click_p2_card(slot: int) -> None:
    """Click Player 2's board card in the specified slot (0‑4, right→left).

    First slot (slot 0) is at x = 60 %, then moves left by 6 % each step.
    """
    if not 0 <= slot <= 4:
        raise ValueError("slot must be between 0 and 4 inclusive")
    rel_x = _P2_BOARD_START_X - _P2_BOARD_STEP_X * slot
    click_relative_to_window(rel_x, _P2_BOARD_Y)

# Hand card helpers remain unchanged

def _hand_rel_x(card_index: int, hand_size: int) -> float:
    """Return the relative x position for a hand card.

    The hand spans horizontally from 5% to 25% of window width. The
    positions are distributed evenly across this 20% range.
    """
    if hand_size < 1:
        raise ValueError("hand_size must be at least 1")
    if not 0 <= card_index < hand_size:
        raise ValueError("card_index must be within the current hand size")

    if hand_size == 1:
        return 0.05  # Single card anchored at the left bound

    step = 0.20 / (hand_size - 1)
    return 0.05 + step * card_index


def click_p1_hand(card_index: int, hand_size: int) -> None:
    """Click a card in Player 1's hand."""
    rel_x = _hand_rel_x(card_index, hand_size)
    click_relative_to_window(rel_x, 0.90)


def click_p2_hand(card_index: int, hand_size: int) -> None:
    """Click a card in Player 2's hand."""
    rel_x = _hand_rel_x(card_index, hand_size)
    click_relative_to_window(rel_x, 0.10)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Demo will click updated board card hotspots in five seconds…")
    time.sleep(5)
    for i in range(5):
        click_p1_card(i)
    click_action0()
    click_end_turn()
    for i in range(5):
        click_p2_card(i)
    click_action0()
    click_end_turn()
    print("Done.")
