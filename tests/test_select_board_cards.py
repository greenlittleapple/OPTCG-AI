"""
Integration Test #4 – Selecting every board card then cancelling.

Sequence (P1 turn):
1–5.  P1 board slots 0-4
6.    Action 0 (Cancel)
7.    End Turn

Sequence (P2 turn):
8–12. P2 board slots 0-4
13.   Action 0 (Cancel)
14.   End Turn
"""

import time
import importlib

GUI = importlib.import_module("utils.gui.gui_automation_starter")


def test_select_board_cards_real():
    GUI.pag.FAILSAFE = True

    time.sleep(2)
    # ---- Player 1 board --------------------------------------------------
    for slot in range(5):                  # slots 0-4 = left→right
        GUI.click_p1_card(slot)
        time.sleep(0.3)

    GUI.click_action0();  time.sleep(0.4)  # Cancel selection
    GUI.click_end_turn(); time.sleep(0.6)  # End P1 turn

    # ---- Player 2 board --------------------------------------------------
    for slot in range(5):                  # slots 0-4 = right→left
        GUI.click_p2_card(slot)
        time.sleep(0.3)

    GUI.click_action0();  time.sleep(0.4)  # Cancel selection
    GUI.click_end_turn(); time.sleep(0.6)  # End P2 turn
