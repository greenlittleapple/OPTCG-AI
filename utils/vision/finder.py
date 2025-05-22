# utils/templates.py
"""Template management + convenience find() wrapper for OPTCG-Sim automation.

Changes in this version
-----------------------
* All logic lives in **TemplateLoader** – easy to instantiate in tests.
* A module-level singleton :data:`loader` preserves the old global behaviour.
* Old helpers (:pydata:`vision`, :pyfunc:`find`, :pyfunc:`load_card`) now
  delegate to that singleton, so no refactor is required elsewhere.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from capture import OPTCGVision

# ---------------------------------------------------------------------------
# File-system layout (adjust if your repo moves)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "vision" / "templates"
BUTTONS_DIR = BASE_DIR / "buttons"  # OPxx-###.png / .jpg live here
CARDS_DIR = BASE_DIR / "cards"  # OPxx-###.png / .jpg live here
LABELS_DIR = BASE_DIR / "labels"  # OPxx-###.png / .jpg live here

CARDS = {
    "OP10-001": CARDS_DIR / "OP10" / "OP10-001.jpg",
    "OP10-004": CARDS_DIR / "OP10" / "OP10-004.jpg",
    "OP10-005": CARDS_DIR / "OP10" / "OP10-005.jpg",
    "OP10-011": CARDS_DIR / "OP10" / "OP10-011.jpg",
    "OP10-016": CARDS_DIR / "OP10" / "OP10-016.jpg",
    "ST21-014": CARDS_DIR / "ST21" / "ST21-014.jpg",
    "OP11-004": CARDS_DIR / "OP11" / "OP11-004.jpg",
    "OP11-017": CARDS_DIR / "OP11" / "OP11-017.jpg",
    "OP01-051": CARDS_DIR / "OP01" / "OP01-051.png",
    "OP05-030": CARDS_DIR / "OP05" / "OP05-030.png",
    "OP06-035": CARDS_DIR / "OP06" / "OP06-035.png",
    "OP10-030": CARDS_DIR / "OP10" / "OP10-030.jpg",
    "OP10-032": CARDS_DIR / "OP10" / "OP10-032.jpg",
    "OP10-018": CARDS_DIR / "OP10" / "OP10-018.jpg",
    "ST21-017": CARDS_DIR / "ST21" / "ST21-017.jpg",
}

BUTTONS = {
    "attack": BUTTONS_DIR / "attack.png",
    "end_turn": BUTTONS_DIR / "end_turn.png",
    "end_turn_2": BUTTONS_DIR / "end_turn_2.png",
    "resolve_attack": BUTTONS_DIR / "resolve_attack.png",
}

STATIC_PATHS = {
    **CARDS,
    **BUTTONS,
}

# ---------------------------------------------------------------------------
# LRU-cached disk loader for card images
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _load_card_from_disk(code: str) -> np.ndarray:
    """
    Read a card template from disk. Path is built from *code*.
    Supports both .png and .jpg.
    """
    # Try PNG first, then JPG/JPEG
    for ext in (".png", ".jpg", ".jpeg"):
        path = CARDS_DIR / f"{code}{ext}"
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    raise FileNotFoundError(f"Card template for {code!r} not found.")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

Match = Tuple[Tuple[int, int], Tuple[int, int], float]  # (top-left), (w,h), score


class TemplateLoader:
    """
    Loads & caches templates and provides a :pyfunc:`find` helper.

    Parameters
    ----------
    static_paths:
        Dict mapping template keys → absolute file paths that should be
        *eagerly* loaded on construction.
    """

    def __init__(self, static_paths: dict[str, Path] | None = None) -> None:
        self._vision = OPTCGVision()
        self._static: dict[str, np.ndarray] = {}

        # Use caller-supplied dict or fallback to module constant
        paths = static_paths if static_paths is not None else STATIC_PATHS
        for key, path in paths.items():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            self._static[key.lower()] = img

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def vision(self) -> OPTCGVision:  # expose capture helper
        return self._vision

    def resolve(self, key: str) -> np.ndarray:
        """
        Return the image for *key* (static or card). Case-insensitive.
        """
        key_lc = key.lower()
        if key_lc in self._static:
            return self._static[key_lc]
        return _load_card_from_disk(key)

    def find(
        self,
        key: str,
        *,
        frame: np.ndarray | None = None,
    ) -> List[Match]:
        """
        Locate all occurrences of *key* in *frame* (or current screen).

        Returns list sorted by descending similarity score.
        """
        template = self.resolve(key)
        if frame is None:
            frame = self._vision.grab()
            if frame is None:
                return []
        hits = OPTCGVision.match_template(frame, template)
        return hits


if __name__ == "__main__":
    loader = TemplateLoader()
    try:
        while True:
            frame = loader.vision.grab()
            hits = loader.find("end_turn", frame=frame)
            for (x, y), (w, h), score in hits:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("OPTCGSim vision test", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cv2.destroyAllWindows()
