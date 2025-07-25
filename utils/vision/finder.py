#!/usr/bin/env python
"""Template management + convenience find() wrapper for OPTCG-Sim automation.

Changes in this version
-----------------------
* All logic lives in **OPTCGVision** – easy to instantiate in tests.
* A module-level singleton :data:`loader` preserves the old global behaviour.
* Old helpers (:pydata:`vision`, :pyfunc:`find`, :pyfunc:`load_card`) now
  delegate to that singleton, so no refactor is required elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.vision.capture import OPTCGVisionHelper

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Portion of the card image to trim from each border before matching
BORDER_PCT = 0.35
# Portion of the card image to keep from the left when loading hand templates
HAND_LEFT_PCT = 0.15

# Scale to convert template size to in-game card size
CARD_SCALE = 99 / 120

# Default similarity threshold for template matching
FIND_THRESHOLD = 0.8

# Cards that should **not** be cropped when loaded. These cards may have
# important details near the border that would be lost otherwise.
UNCROPPED_CARDS = {"DON_side_p1", "DON_side_p2"}

__all__ = ["OPTCGVision", "loader", "find", "load_card"]

# ---------------------------------------------------------------------------
# Helper: crop card borders
# ---------------------------------------------------------------------------


def _crop_card_border(img: np.ndarray) -> np.ndarray:
    """Return the image cropped for template matching."""
    h, w = img.shape[:2]
    dx, dy = int(w * BORDER_PCT), int(h * BORDER_PCT)
    cropped = img[dy : h - dy, dx : w - dx]
    return cropped


def _left_edge(img: np.ndarray) -> np.ndarray:
    """Return only the leftmost portion of a card image for hand detection."""
    h, w = img.shape[:2]
    width = int(w * HAND_LEFT_PCT)
    return img[:, :width]


# ---------------------------------------------------------------------------
# File-system layout (adjust if your repo moves)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "vision" / "templates"
UNSCALED_DIR = BASE_DIR / "unscaled"  # Unscaled template images live here
CARDS_DIR = BASE_DIR / "cards"  # OPxx-###.png / .jpg live here

# Automatically map card IDs to their template paths. The files currently use
# the naming scheme "<card-id>_small.<ext>"; strip the suffix to obtain the
# card code used throughout the project.
CARDS = {
    (p.stem[:-6] if p.stem.endswith("_small") else p.stem): p
    for p in CARDS_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
}

# Automatically map unscaled template names to their paths. Any image file
# found in ``UNSCALED_DIR`` becomes a key using its stem.
UNSCALED = {
    p.stem: p
    for p in UNSCALED_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
}

STATIC_PATHS = {
    **CARDS,
    **UNSCALED,
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
                if code not in UNCROPPED_CARDS:
                    img = _crop_card_border(img)
                return img
    raise FileNotFoundError(f"Card template for {code!r} not found.")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

Match = Tuple[Tuple[int, int], Tuple[int, int], float]  # (top-left), (w,h), score


@dataclass
class OPTCGObs:
    can_attack: bool
    can_blocker: bool
    can_choose_from_top: bool
    can_choose_friendly_target: bool
    can_choose_enemy_target: bool
    can_deploy: bool
    can_draw: bool
    can_end_turn: bool
    can_resolve: bool
    can_return_cards: bool
    choice_cards: List[str]
    hand_p1: List[str]
    hand_p2: List[str]
    board_p1: List[str]
    board_p2: List[str]
    rested_cards_p1: List[int]
    rested_cards_p2: List[int]
    num_active_don_p1: int
    num_active_don_p2: int
    num_life_p1: int
    num_life_p2: int


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
        self._hand_templates: dict[str, np.ndarray] = {}
        self._board_templates: dict[str, np.ndarray] = {}

        # Use caller-supplied dict or fallback to module constant
        paths = static_paths if static_paths is not None else STATIC_PATHS
        for key, path in paths.items():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            key_lc = key.lower()
            if key in CARDS:
                board_img = img if key in UNCROPPED_CARDS else _crop_card_border(img)
                if key not in UNCROPPED_CARDS:
                    hand_img = _left_edge(img)
                    self._hand_templates[key_lc] = hand_img
                self._board_templates[key_lc] = board_img
                self._static[key_lc] = board_img
            else:
                self._static[key_lc] = img

    def _show_debug(self, roi: np.ndarray):
        cv2.imshow("debug", roi)
        cv2.waitKey()

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

    def resolve(self, key: str, *, hand: bool = False) -> np.ndarray:
        """
        Return the image for *key* (static or card). Case-insensitive.

        The ``hand`` flag selects the cropped template used for hand scanning.
        """
        key_lc = key.lower()
        if key_lc in self._static:
            if key_lc in self._hand_templates:
                return (
                    self._hand_templates[key_lc]
                    if hand
                    else self._board_templates[key_lc]
                )
            return self._static[key_lc]
        return _load_card_from_disk(key)

    def find(
        self,
        key: str,
        frame: np.ndarray | None = None,
        is_card: bool = False,
        rotated: bool = False,
        *,
        hand: bool = False,
    ) -> List[Match]:
        """
        Locate all occurrences of *key* in *frame* (or current screen).

        Returns list sorted by descending similarity score.
        """
        template = self.resolve(key, hand=hand)
        if frame is None:
            frame = self.grab()
            if frame is None:
                return []
        threshold = 0.9 if hand else FIND_THRESHOLD
        scales = CARD_SCALE if is_card else 1.0
        if rotated:
            template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
        hits = OPTCGVisionHelper.match_template(
            frame, template, threshold=threshold, scales=scales
        )
        return hits

    def _detect_card_in_roi(
        self, roi: np.ndarray, rotated: bool = False, *, hand: bool = False
    ) -> str:
        """Return the first card template that matches in `roi`, else None."""
        for name in CARDS:  # simple linear scan
            if hand and name in UNCROPPED_CARDS:
                continue
            if self.find(name, frame=roi, is_card=True, rotated=rotated, hand=hand):
                return name
        return ""

    def _detect_card_and_rest(self, roi: np.ndarray) -> Tuple[str, bool]:
        """Return (card, rested) for the first matching card in `roi`."""
        # Try upright orientation first
        card = self._detect_card_in_roi(roi)
        if card:
            return card, False

        # Try rotated 90 degrees clockwise (rested)
        card = self._detect_card_in_roi(roi, rotated=True)
        if card:
            return card, True

        return "", False

    def scan(self) -> OPTCGObs:
        """Capture a frame and return high-level observations.

        The function scans player hands, board slots and other button cues.

        Returns:
            Observation dict containing board state, hand contents and button
            cues.
        """
        frame = self.grab()
        h, w = frame.shape[:2]

        # 1. Button cues -----------------------------------------------------
        btn_y0, btn_y1 = int(0.50 * h), h
        btn_x0, btn_x1 = int(0.70 * w), w
        button_area = frame[btn_y0:btn_y1, btn_x0:btn_x1]
        buttons = {name: self.find(name, frame=button_area) for name in UNSCALED}
        can_choose_from_top = bool(buttons.get("choose_0_targets")) or bool(
            buttons.get("choose_-1_targets")
        )
        can_draw = bool(buttons.get("dont_draw_any"))

        # 2. Constants -------------------------------------------------------
        SLOT_WIDTH_PCT = 0.03
        SLOTS = 5
        HAND_TOTAL_WIDTH_PCT = 0.2
        HAND_SCAN_X0 = 0.0
        HAND_SCAN_X1 = 0.25
        HAND_SLOT_START_PCT = 0.02
        HAND_MAX_SIZE = 10
        CHOICE_SHIFT_PCT = 0.05
        BOARD_WIDTH_PCT, BOARD_STEP_PCT = 0.10, 0.06
        BOARD_P1_START_X, BOARD_P1_Y = 0.40, 0.6
        BOARD_P2_START_X, BOARD_P2_Y = 0.60, 0.45
        BOARD_HEIGHT_PCT = 0.20
        DON_P1_START_X, DON_P1_END_X, DON_P1_Y = 0.35, 0.6, 0.90
        DON_P2_START_X, DON_P2_END_X, DON_P2_Y = 0.65, 0.4, 0.15
        DON_HEIGHT_PCT = 0.20
        LIFE_P1_X0_PCT, LIFE_P1_Y0_PCT, LIFE_P1_X1_PCT, LIFE_P1_Y1_PCT = (
            0.30,
            0.55,
            0.40,
            0.80,
        )
        LIFE_P2_X0_PCT, LIFE_P2_Y0_PCT, LIFE_P2_X1_PCT, LIFE_P2_Y1_PCT = (
            0.60,
            0.20,
            0.70,
            0.45,
        )
        LIFE_SCAN_PCT = 0.10

        def count_life_cards(
            x0_pct: float,
            y0_pct: float,
            x1_pct: float,
            y1_pct: float,
            *,
            bottom: bool,
        ) -> int:
            """Return the number of life cards in the specified region."""
            x0 = int(x0_pct * w)
            x1 = int(x1_pct * w)
            y0 = int(y0_pct * h)
            y1 = int(y1_pct * h)
            roi = frame[y0:y1, x0:x1]

            template = self.resolve("card_back")
            tpl_h = int(template.shape[0] * LIFE_SCAN_PCT)
            if bottom:
                template = template[template.shape[0] - tpl_h :, :]
            else:
                template = template[:tpl_h, :]

            hits = OPTCGVisionHelper.match_template(
                roi, template, threshold=FIND_THRESHOLD, scales=1.0
            )
            return len(hits)

        def count_hand_cards(y0: int, y1: int) -> int:
            x0 = int(HAND_SCAN_X0 * w)
            x1 = int(HAND_SCAN_X1 * w)
            roi = frame[y0:y1, x0:x1]
            total = 0
            for name in CARDS:
                if name in UNCROPPED_CARDS:
                    continue
                total += len(self.find(name, frame=roi, is_card=True, hand=True))
            return total

        def scan_hand(y0: int, y1: int, hand_size: int) -> List[str]:
            cards: List[str] = []
            if hand_size == 0:
                return cards
            shift_pct = HAND_TOTAL_WIDTH_PCT / max(hand_size - 1, 4)
            for i in range(hand_size):
                x0 = int((HAND_SLOT_START_PCT + shift_pct * i) * w)
                x1 = int(x0 + SLOT_WIDTH_PCT * w)
                roi = frame[y0:y1, x0:x1]
                cards.append(self._detect_card_in_roi(roi, hand=True))
            cards += [''] * (HAND_MAX_SIZE - len(cards))
            return cards

        def scan_board(
            start_x: float, step_x: float, y_center: float
        ) -> Tuple[List[str], List[int]]:
            y0 = int((y_center - BOARD_HEIGHT_PCT / 2) * h)
            y1 = int((y_center + BOARD_HEIGHT_PCT / 2) * h)
            slots: List[str] = []
            rested: List[int] = []
            for i in range(SLOTS):
                center_x = start_x + step_x * i
                x0 = int((center_x - BOARD_WIDTH_PCT / 2) * w)
                x1 = int((center_x + BOARD_WIDTH_PCT / 2) * w)
                roi = frame[y0:y1, x0:x1]
                card, is_rest = self._detect_card_and_rest(roi)
                slots.append(card)
                rested.append(int(is_rest))
            return slots, rested

        def scan_choices(y0: int, y1: int) -> List[str]:
            """Scan up to 5 selectable cards arranged like the P1 hand."""
            cards: List[str] = []
            for i in range(SLOTS):
                x0 = int((HAND_SLOT_START_PCT + CHOICE_SHIFT_PCT * i) * w)
                x1 = int(x0 + SLOT_WIDTH_PCT * w)
                roi = frame[y0:y1, x0:x1]
                cards.append(self._detect_card_in_roi(roi, hand=True))
            return cards

        def scan_don(
            start_x: float, end_x: float, y_center: float, template: str
        ) -> int:
            """Return the number of active DON cards in the specified row."""
            y0 = int((y_center - DON_HEIGHT_PCT / 2) * h)
            y1 = int((y_center + DON_HEIGHT_PCT / 2) * h)
            x0 = int(min(start_x, end_x) * w)
            x1 = int(max(start_x, end_x) * w)
            roi = frame[y0:y1, x0:x1]
            hits = self.find(template, frame=roi, is_card=True)
            return min(len(hits), 10)

        # 3. Player-1 --------------------------------------------------------
        p1_y0, p1_y1 = int(0.80 * h), h
        p1_count = count_hand_cards(p1_y0, p1_y1)
        hand_p1 = scan_hand(p1_y0, p1_y1, p1_count)
        board_p1, rested_p1 = scan_board(BOARD_P1_START_X, BOARD_STEP_PCT, BOARD_P1_Y)
        num_life_p1 = count_life_cards(
            LIFE_P1_X0_PCT,
            LIFE_P1_Y0_PCT,
            LIFE_P1_X1_PCT,
            LIFE_P1_Y1_PCT,
            bottom=True,
        )

        # 4. Player-2 --------------------------------------------------------
        p2_y0, p2_y1 = 0, int(0.20 * h)
        p2_count = count_hand_cards(p2_y0, p2_y1)
        hand_p2 = scan_hand(p2_y0, p2_y1, p2_count)
        board_p2, rested_p2 = scan_board(BOARD_P2_START_X, -BOARD_STEP_PCT, BOARD_P2_Y)
        num_life_p2 = count_life_cards(
            LIFE_P2_X0_PCT,
            LIFE_P2_Y0_PCT,
            LIFE_P2_X1_PCT,
            LIFE_P2_Y1_PCT,
            bottom=False,
        )

        num_active_don_p1 = scan_don(
            DON_P1_START_X, DON_P1_END_X, DON_P1_Y, "DON_side_p1"
        )
        num_active_don_p2 = scan_don(
            DON_P2_START_X, DON_P2_END_X, DON_P2_Y, "DON_side_p2"
        )

        # 5. Choice row ------------------------------------------------------
        choice_cards: List[str] = ["", "", "", "", ""]
        if can_choose_from_top or can_draw:
            choice_y0, choice_y1 = int(0.65 * h), int(0.85 * h)
            choice_cards = scan_choices(choice_y0, choice_y1)

        # 6. Pack observations ----------------------------------------------
        obs = OPTCGObs(
            can_attack = bool(buttons.get("attack")),
            can_blocker = bool(buttons.get("no_blocker")),
            can_choose_from_top = can_choose_from_top,
            can_choose_friendly_target = bool(buttons.get("choose_0_friendly_targets")) or bool(buttons.get("select_character_to_replace")),
            can_choose_enemy_target = bool(buttons.get("select_target")),
            can_deploy = bool(buttons.get("deploy")),
            can_draw = can_draw,
            can_end_turn = bool(buttons.get("end_turn")),
            can_resolve = bool(buttons.get("resolve_attack")),
            can_return_cards = bool(buttons.get("return_cards_to_deck")),
            hand_p1 = hand_p1,
            hand_p2 = hand_p2,
            board_p1 = board_p1,
            board_p2 = board_p2,
            rested_cards_p1 = rested_p1,
            rested_cards_p2 = rested_p2,
            num_active_don_p1 = num_active_don_p1,
            num_active_don_p2 = num_active_don_p2,
            num_life_p1 = num_life_p1,
            num_life_p2 = num_life_p2,
            choice_cards = choice_cards,
        )

        return obs


# ---------------------------------------------------------------------------
# Module-level convenience helpers -----------------------------------------
# ---------------------------------------------------------------------------

# Global singleton for convenience APIs
loader = OPTCGVision()


def find(
    key: str,
    frame: np.ndarray | None = None,
    is_card: bool = False,
    *,
    hand: bool = False,
) -> List[Match]:
    """Module-level helper that delegates to :data:`loader`."""
    return loader.find(key, frame=frame, is_card=is_card, hand=hand)


def load_card(code: str) -> np.ndarray:
    """Return the template image for *code* using the module loader."""
    return loader.resolve(code)


def test_find(key: str):
    vision = OPTCGVision()
    try:
        while True:
            frame = vision.grab()
            hits = vision.find(key, frame=frame, is_card=key in CARDS.keys())
            hits += vision.find(
                key, frame=frame, is_card=key in CARDS.keys(), rotated=True
            )
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
    # test_find("OP08-013")
    vision = OPTCGVision()
    try:
        while True:
            frame = vision.grab()
            obs = vision.scan()
            pprint(obs)
            cv2.imshow("OPTCGSim vision test", frame)
            cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
