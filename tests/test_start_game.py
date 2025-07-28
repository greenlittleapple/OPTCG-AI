"""
Integration Test #1 – Starting a game through the real GUI.

Sequence:
1. Solo vs Self   (click_solo_vs_self)
2. Start          (click_start)
3. Mulligan P1    (click_action1)
4. Mulligan P2    (click_action0)
5. End Turn P1    (click_end_turn)
6. End Turn P2    (click_end_turn)

The test passes if all calls execute without raising an exception.
"""

import time
import importlib

GUI = importlib.import_module("utils.gui.gui_automation_starter")


def test_start_game_real():
    # Small safety: move mouse to top-left to abort mis-click loop.
    GUI.pag.FAILSAFE = True

    time.sleep(2)
    GUI.click_solo_vs_self(); time.sleep(0.6)
    GUI.click_start();        time.sleep(0.6)
    GUI.click_action1();      time.sleep(0.4)   # P1 mulligan
    GUI.click_action0();      time.sleep(0.4)   # P2 mulligan
    MACROS.end_turn();     time.sleep(0.5)   # P1
    MACROS.end_turn();     time.sleep(0.5)   # P2

    # No assertion needed: we only care that no error was raised.
