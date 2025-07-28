"""
Integration Test #8 – Deploying a hand card using the macro.

Sequence:
1. Player 1 deploys hand card #0 (Action 1 to confirm)
2. End Turn
3. Player 2 deploys hand card #0
4. End Turn
"""

import time
import importlib

GUI    = importlib.import_module("utils.gui.gui_automation_starter")
MACROS = importlib.import_module("utils.gui.gui_macros")


def test_deploy_card_real():
    GUI.pag.FAILSAFE = True

    time.sleep(2)
    MACROS.deploy_card(acting_player=1, hand_card_index=0, hand_size=6)
    time.sleep(0.5)
    MACROS.end_turn(); time.sleep(0.7)

    MACROS.deploy_card(acting_player=2, hand_card_index=0, hand_size=7)
    time.sleep(0.5)
    MACROS.end_turn(); time.sleep(0.7)
    # No assertion required; any exception would fail the test.
