"""
Integration Test #2 – Leader-vs-Leader attack (updated API).

Sequence (no extra card-selection clicks needed because `attack()` now
handles them internally):

P1 turn
--------
1. P1 leader attacks P2 leader            (MACROS.attack)
2. P2 resolves attack                     (click_action0)
3. End Turn                               (click_end_turn)

P2 turn
--------
4. P2 leader attacks P1 leader            (MACROS.attack)
5. P1 resolves attack                     (click_action0)
6. End Turn                               (click_end_turn)
"""

import time
import importlib

GUI    = importlib.import_module("utils.gui.gui_automation_starter")
MACROS = importlib.import_module("utils.gui.gui_macros")


def test_leader_attack_real():
    GUI.pag.FAILSAFE = True

    time.sleep(2)
    # ---- Player 1 turn --------------------------------------------------
    MACROS.attack(acting_player=0, acting_card_index=0,
                  target_player=1, target_card_index=0)
    time.sleep(0.4)
    GUI.click_action0();      time.sleep(0.4)   # P2 resolves
    MACROS.end_turn();     time.sleep(0.6)

    # ---- Player 2 turn --------------------------------------------------
    MACROS.attack(acting_player=1, acting_card_index=0,
                  target_player=0, target_card_index=0)
    time.sleep(0.4)
    GUI.click_action0();      time.sleep(0.4)   # P1 resolves
    MACROS.end_turn();     time.sleep(0.6)

    # If we reach here without an exception, the test passes.
