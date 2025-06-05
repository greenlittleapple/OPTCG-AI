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
    • DON counters: generic selector for the 10 DON cards

*Board‑card spacing update*
---------------------------
P1 board slots are spaced **6 %** apart (was 5 %).
First slot still starts at x = 40 %.

P2 board slots mirror this, but the entire row shifts **10 % leftward**
so the right‑most slot now starts at x = 60 %.

*DON selector*
--------------
DON counters are arranged horizontally from x = 40 % to 60 % at y = 90 %.
Index 0 is the left‑most DON; index 9 is the right‑most.

Dependencies:
    pywin32    (preferred for cursorless clicks on Windows)
    pyautogui  (fallback when pywin32 is unavailable)
    pygetwindow

Adjust the WINDOW_TITLE constant if the client window has a different title.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import pyautogui as pag
import pygetwindow as gw  # type: ignore
try:
    import win32api
    import win32con
    import win32gui
except Exception:  # pragma: no cover - platform-specific optional deps
    win32api = win32con = win32gui = None

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


def _send_click(win: gw.Window, x: int, y: int) -> None:
    """Send a left-click to *win* at absolute screen position ``(x, y)``.

    Uses :mod:`pywin32` if available to avoid moving the real cursor. The
    function first issues ``WM_MOUSEMOVE`` followed by ``WM_LBUTTONDOWN`` and
    ``WM_LBUTTONUP`` via :func:`win32gui.SendMessage`.

    On non-Windows platforms (or if pywin32 is not installed) this falls back
    to :func:`pyautogui.click`.
    """
    if win32gui is None:
        pag.click(x, y)
        return

    client_x, client_y = win32gui.ScreenToClient(win._hWnd, (x, y))
    lparam = win32api.MAKELONG(client_x, client_y)
    win32gui.PostMessage(win._hWnd, win32con.WM_MOUSEMOVE, 0, lparam)
    win32gui.PostMessage(win._hWnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    time.sleep(0.01)
    win32gui.PostMessage(win._hWnd, win32con.WM_LBUTTONUP, 0, lparam)


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
    _send_click(win, x, y)
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

    Spacing between slots is 6 % of window width.
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

# Hand card helpers

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
# DON card helpers  ----------------------------------------------------------
# P1 row (unchanged)

_P1_DON_START_X = 0.40
_P1_DON_END_X   = 0.55
_P1_DON_Y       = 0.90
_P1_DON_STEP_X  = (_P1_DON_END_X - _P1_DON_START_X) / 10


def click_don(don_index: int) -> None:
    """Backward-compat alias: same as click_p1_don()."""
    click_p1_don(don_index)


def click_p1_don(don_index: int) -> None:
    """Click one of Player 1's 10 DON counters (index 0 = left-most)."""
    if not 0 <= don_index <= 9:
        raise ValueError("don_index must be between 0 and 9 inclusive")
    rel_x = _P1_DON_START_X + _P1_DON_STEP_X * don_index
    click_relative_to_window(rel_x, _P1_DON_Y)

# --- NEW: P2 row -----------------------------------------------------------

_P2_DON_START_X = 0.60          # right-most DON (index 0)
_P2_DON_END_X   = 0.45          # left-most DON  (index 9)
_P2_DON_Y       = 0.10
_P2_DON_STEP_X  = (_P2_DON_END_X - _P2_DON_START_X) / 10  # negative step


def click_p2_don(don_index: int) -> None:
    """Click one of Player 2's 10 DON counters (index 0 = right-most)."""
    if not 0 <= don_index <= 9:
        raise ValueError("don_index must be between 0 and 9 inclusive")
    rel_x = _P2_DON_START_X + _P2_DON_STEP_X * don_index
    click_relative_to_window(rel_x, _P2_DON_Y)
