# utils/vision/finder.py
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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from capture import OPTCGVisionHelper

# ---------------------------------------------------------------------------
# File-system layout (adjust if your repo moves)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "vision" / "templates"
BUTTONS_DIR = BASE_DIR / "buttons"  # OPxx-###.png / .jpg live here
CARDS_DIR = BASE_DIR / "cards"  # OPxx-###.png / .jpg live here
LABELS_DIR = BASE_DIR / "labels"  # OPxx-###.png / .jpg live here

# Automatically map card IDs to their template paths. The files currently use
# the naming scheme "<card-id>_small.<ext>"; strip the suffix to obtain the
# card code used throughout the project.
CARDS = {
    (p.stem[:-6] if p.stem.endswith("_small") else p.stem): p
    for p in CARDS_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
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


class OPTCGVision:
    """
    Loads & caches templates and provides a :pyfunc:`find` helper.

    Parameters
    ----------
    static_paths:
        Dict mapping template keys → absolute file paths that should be
        *eagerly* loaded on construction.
    """

    def __init__(self, static_paths: dict[str, Path] | None = None) -> None:
        self._helper = OPTCGVisionHelper()
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
    def helper(self) -> OPTCGVisionHelper:  # expose capture helper
        return self._helper

    def grab(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Screen grab of OPTCGSim
        """
        return self._helper.grab()

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
        frame: np.ndarray | None = None,
        is_card: bool = False,
    ) -> List[Match]:
        """
        Locate all occurrences of *key* in *frame* (or current screen).

        Returns list sorted by descending similarity score.
        """
        template = self.resolve(key)
        if frame is None:
            frame = self.grab()
            if frame is None:
                return []
        threshold = 0.95
        scales = 99/120 if is_card else (1.0) # scale to (in-game card size / card template size)
        hits = OPTCGVisionHelper.match_template(
            frame, template, threshold=threshold, scales=scales
        )
        return hits

    def _detect_card_in_roi(self, roi: np.ndarray) -> str:
        """Return the first card template that matches in `roi`, else None."""
        for name in CARDS:  # simple linear scan
            if self.find(name, frame=roi, is_card=True):
                return name
        return ""

    def scan(self, include_initial_hands: bool = False) -> Dict[str, Any]:
        """Capture a frame and return high-level observations.

        Args:
            include_initial_hands: If True, scan all five hand slots for each
            player and return `initial_hand_p1/p2`.  If False, skip that work.

        The function also scans each board slot for both players using preset
        coordinates.

        Returns:
            Observation dict.  Initial-hand keys appear only if requested.
            Board state is always included under ``board_p1`` and ``board_p2``.
        """
        frame = self.grab()
        h, w = frame.shape[:2]

        # 1. Button cues -----------------------------------------------------
        btn_y0, btn_y1 = int(0.50 * h), h
        btn_x0, btn_x1 = int(0.70 * w), w
        cropped_buttons = frame[btn_y0:btn_y1, btn_x0:btn_x1]
        buttons = {name: self.find(name, frame=cropped_buttons) for name in BUTTONS}

        # 2. Constants -------------------------------------------------------
        SLOT_WIDTH_PCT, SLOT_SHIFT_PCT, SLOTS = 0.10, 0.05, 5
        BOARD_WIDTH_PCT, BOARD_STEP_PCT = 0.10, 0.06
        BOARD_P1_START_X, BOARD_P1_Y = 0.40, 0.6
        BOARD_P2_START_X, BOARD_P2_Y = 0.60, 0.45
        BOARD_HEIGHT_PCT = 0.20

        def scan_hand(y0: int, y1: int, ordered: bool):
            """Either return list of 5 slots (ordered=True) or only newest slot."""
            if ordered:
                cards: List[Optional[str]] = []
                for i in range(SLOTS):
                    x0 = int(SLOT_SHIFT_PCT * i * w)
                    x1 = int(x0 + SLOT_WIDTH_PCT * w)
                    roi = frame[y0:y1, x0:x1]
                    cards.append(self._detect_card_in_roi(roi))
                return cards
            else:
                x0 = int(SLOT_SHIFT_PCT * 4 * w)
                x1 = int(x0 + SLOT_WIDTH_PCT * w)
                roi = frame[y0:y1, x0:x1]
                return self._detect_card_in_roi(roi)

        def scan_board(
            start_x: float, step_x: float, y_center: float, right_to_left: bool = False
        ) -> List[str]:
            y0 = int((y_center - BOARD_HEIGHT_PCT / 2) * h)
            y1 = int((y_center + BOARD_HEIGHT_PCT / 2) * h)
            slots: List[str] = []
            for i in range(SLOTS):
                center_x = start_x + step_x * i
                x0 = int((center_x - BOARD_WIDTH_PCT / 2) * w)
                x1 = int((center_x + BOARD_WIDTH_PCT / 2) * w)
                roi = frame[y0:y1, x0:x1]
                # cv2.imshow("board", roi)
                # cv2.waitKey(0)
                slots.append(self._detect_card_in_roi(roi))
            return slots[::-1] if right_to_left else slots

        # 3. Player-1 --------------------------------------------------------
        p1_y0, p1_y1 = int(0.80 * h), h
        if include_initial_hands:
            initial_hand_p1 = scan_hand(p1_y0, p1_y1, True)
            latest_card_p1 = initial_hand_p1[4]
        else:
            latest_card_p1 = scan_hand(p1_y0, p1_y1, False)
            initial_hand_p1 = None
        board_p1 = scan_board(BOARD_P1_START_X, BOARD_STEP_PCT, BOARD_P1_Y)

        # 4. Player-2 --------------------------------------------------------
        p2_y0, p2_y1 = 0, int(0.20 * h)
        if include_initial_hands:
            initial_hand_p2 = scan_hand(p2_y0, p2_y1, True)
            latest_card_p2 = initial_hand_p2[4]
        else:
            latest_card_p2 = scan_hand(p2_y0, p2_y1, False)
            initial_hand_p2 = None
        board_p2 = scan_board(
            BOARD_P2_START_X, -BOARD_STEP_PCT, BOARD_P2_Y, right_to_left=True
        )

        # 5. Pack observations ----------------------------------------------
        obs: Dict[str, Any] = {
            "can_attack": bool(buttons.get("attack")),
            "can_resolve": bool(buttons.get("resolve_attack")),
            "can_end_turn": bool(buttons.get("end_turn")),
            "latest_card_p1": latest_card_p1,
            "latest_card_p2": latest_card_p2,
            "board_p1": board_p1,
            "board_p2": board_p2,
        }
        if include_initial_hands:
            obs["initial_hand_p1"] = initial_hand_p1
            obs["initial_hand_p2"] = initial_hand_p2

        return obs


def test_find(key: str):
    vision = OPTCGVision()
    try:
        while True:
            frame = vision.grab()
            hits = vision.find(key, frame=frame, is_card=key in CARDS.keys())
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


if __name__ == "__main__":
    test_find("OP08-010")
    # vision = OPTCGVision()
    # try:
    #     while True:
    #         frame = vision.grab()
    #         obs = vision.scan(True)
    #         print(obs)
    #         cv2.imshow("OPTCGSim vision test", frame)
    #         cv2.waitKey(0)
    # finally:
    #     cv2.destroyAllWindows()
