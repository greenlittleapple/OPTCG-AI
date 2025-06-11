"""
Integration Test #7 – Clicking a random choice card from the selection row.

The test simply clicks one of the choice cards and exits. The specific
card is chosen at random to mimic human input.
"""

import time
import random
import importlib

GUI = importlib.import_module("utils.gui.gui_automation_starter")


def test_click_random_choice_card():
    GUI.pag.FAILSAFE = True

    time.sleep(2)
    idx = random.randint(0, 4)
    GUI.click_choice_card(idx)
    # Test passes if no exception is raised.
