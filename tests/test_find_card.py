"""
Integration Test #6 – Detecting card **OP10-001** on screen.

Pre-conditions
--------------
1. OPTCGSim.exe is running.
2. The card with ID **OP10-001** is fully visible somewhere in the current
   game window (hand, board, trash, or zoom view).
3. `assets/templates/cards/OP10-001.png` exists.

Expected result
---------------
`utils.templates.find("OP10-001")` returns **≥ 1** match.
"""

import time
import importlib

# Dynamic import keeps path assumptions identical to other integration tests
TEMPLATES = importlib.import_module("utils.vision.finder")

def test_find_card():
    """
    Try up to 5 consecutive frames to catch the card; fail if none are found.
    """
    finder = TEMPLATES.TemplateLoader()
    attempts, hits = 5, []
    for _ in range(attempts):
        hits = finder.find("OP10-001")
        if hits:      # at least one (pt, size, score) tuple returned
            break
        time.sleep(0.2)          # small delay for next frame

    assert hits, "OP10-001 card not detected on screen."
