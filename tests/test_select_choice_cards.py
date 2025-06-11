"""
Integration Test #7 – Clicking all choice cards in the selection row.

The test simply clicks each of the five temporary choice cards once in
order to verify their relative positions.
"""

import time
import importlib

GUI = importlib.import_module("utils.gui.gui_automation_starter")


def test_select_choice_cards():
    GUI.pag.FAILSAFE = True

    time.sleep(2)
    for idx in range(5):
        GUI.click_choice_card(idx)
        time.sleep(0.3)
    # Test passes if no exception is raised.
