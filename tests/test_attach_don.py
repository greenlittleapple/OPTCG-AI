"""
Integration Test #5 – Comprehensive DON!! attachment scenarios.

Sequence:
1.  P1 attaches 10 DON to board card slot-1
2.  End Turn
3.  P2 attaches 10 DON to board card slot-1
4.  End Turn
5.  P1 has 7 total DON (3 attachable); attaches 2 DON to its Leader
6.  End Turn
7.  P2 has 5 total DON (all 5 attachable); attaches 3 DON to its Leader
8.  End Turn
"""

import time
import importlib

GUI    = importlib.import_module("utils.gui.gui_automation_starter")
MACROS = importlib.import_module("utils.gui.gui_macros")


def test_attach_don_real():
    GUI.pag.FAILSAFE = True
    time.sleep(2)  # allow the game window to fully load

    # ---- Phase 1: attach 10 DON to board card #1 for both players --------
    MACROS.attach_don(
        acting_player=0,
        card_index=1,   # board card #1
        attachable_don=10,
        total_don=10,
        num_to_attach=10
    )
    time.sleep(0.6)
    MACROS.end_turn(); time.sleep(0.8)

    MACROS.attach_don(
        acting_player=1,
        card_index=1,   # board card #1
        attachable_don=10,
        total_don=10,
        num_to_attach=10
    )
    time.sleep(0.6)
    MACROS.end_turn(); time.sleep(0.8)

    # ---- Phase 2: partial DON attachment to leaders ----------------------
    # P1: 7 total DON, 3 attachable, attach only 2 to the Leader
    MACROS.attach_don(
        acting_player=0,
        card_index=0,   # Leader
        attachable_don=3,
        total_don=7,
        num_to_attach=2
    )
    time.sleep(0.6)
    MACROS.end_turn(); time.sleep(0.8)

    # P2: 5 total DON, 5 attachable, attach 3 to the Leader
    MACROS.attach_don(
        acting_player=1,
        card_index=0,   # Leader
        attachable_don=5,
        total_don=5,
        num_to_attach=3
    )
    time.sleep(0.6)
    MACROS.end_turn(); time.sleep(0.8)

    # Test passes if no exception is raised.
