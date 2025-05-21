"""
Integration Test #3 – Selecting every card in hand (hand size = 5)

Sequence (P1 turn):
1–5.  click_p1_hand(0-4, 5)
6.    click_action0()   # Cancel / close hand view
7.    click_end_turn()

Sequence (P2 turn):
8–12. click_p2_hand(0-4, 5)
13.   click_action0()   # Cancel
14.   click_end_turn()

The test passes if the clicks execute without raising an exception.
"""

import time
import importlib

GUI = importlib.import_module("utils.gui.gui_automation_starter")


def test_select_hand_cards():
    # Enable pyautogui failsafe (mouse to top-left aborts test)
    GUI.pag.FAILSAFE = True

    time.sleep(2)
    # ---- Player 1 hand ----------------------------------------------------
    for idx in range(6):
        GUI.click_p1_hand(card_index=idx, hand_size=6)
        time.sleep(0.3)

    GUI.click_action0(); time.sleep(0.4)   # Cancel
    GUI.click_end_turn(); time.sleep(0.6)  # End P1 turn

    # ---- Player 2 hand ----------------------------------------------------
    for idx in range(7):
        GUI.click_p2_hand(card_index=idx, hand_size=7)
        time.sleep(0.3)

    GUI.click_action0(); time.sleep(0.4)   # Cancel
    GUI.click_end_turn(); time.sleep(0.6)  # End P2 turn

    # No explicit assertions—test passes if no exception occurs.
