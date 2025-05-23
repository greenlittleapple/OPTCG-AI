# utils/capture.py
"""OPTCG‑Sim computer‑vision helpers
====================================
Capture frames directly from the **OPTCGSim.exe** window—even when other
windows cover it—and offer thin wrappers around OpenCV template matching.

### Installation
```
pip install opencv-python numpy psutil pywin32 mss
```
Python 3.9+ is recommended.

The module has **zero runtime dependencies on PyAutoGUI**; you can still use
those APIs elsewhere, but they are intentionally decoupled here so vision tests
can run headless.
"""
from __future__ import annotations

import ctypes
import enum
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import mss  # type: ignore
import numpy as np  # type: ignore
import psutil  # type: ignore
import win32con  # type: ignore
import win32gui  # type: ignore
import win32process  # type: ignore
import win32ui  # type: ignore
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:/Programs/Tesseract-OCR/tesseract.exe"
# ---------------------------------------------------------------------------
# Internal helpers – Win32 plumbing
# ---------------------------------------------------------------------------

_PW_RENDERFULLCONTENT = 0x00000002  # Capture even if occluded (≥ Win 8)


def _enum_hwnds_for_pid(pid: int) -> List[int]:
    """Return all top‑level windows belonging to *pid*."""

    def _callback(hwnd: int, hwnds: List[int]):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            _, this_pid = win32process.GetWindowThreadProcessId(hwnd)
            if this_pid == pid:
                hwnds.append(hwnd)
        return True

    collected: List[int] = []
    win32gui.EnumWindows(_callback, collected)
    return collected


def _get_pid_by_name(exe_name: str) -> Optional[int]:
    for proc in psutil.process_iter(["name"]):
        if proc.info.get("name", "").lower() == exe_name.lower():
            return proc.pid  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# Capture back‑end
# ---------------------------------------------------------------------------


class CaptureBackend(enum.Enum):
    """Which Windows API to grab pixels with."""

    PRINTWINDOW = "printwindow"  # Win32 PrintWindow (off‑screen capable)
    MSS_MONITOR = "mss_monitor"  # Fallback – full‑screen grab via DXGI


class CaptureError(RuntimeError):
    """Raised when a frame could not be captured."""


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------


class OPTCGVisionHelper:
    """High‑level helper to capture frames & run template matching.

    Parameters
    ----------
    exe_name:
        Executable name of the running sim (default: ``"OPTCGSim.exe"``).
    backend:
        • ``CaptureBackend.PRINTWINDOW`` (default) – captures the *window* only
          and still works if it is hidden behind other windows (not minimised).
        • ``CaptureBackend.MSS_MONITOR`` – captures the entire monitor the first
          time and crops the sim window rectangle from that. Slightly faster on
          high‑refresh monitors but **cannot** see occluded content.
    hwnd:
        If you already know the window handle you want, pass it here and both
        the process lookup & EnumWindows pass are skipped.
    """

    def __init__(
        self,
        exe_name: str = "OPTCGSim.exe",
        backend: CaptureBackend = CaptureBackend.PRINTWINDOW,
        hwnd: Optional[int] = None,
    ) -> None:
        self.exe_name = exe_name
        self.backend = backend
        self._hwnd: Optional[int] = hwnd or self._find_main_window()
        if self._hwnd is None:
            raise CaptureError(f"No visible window found for {exe_name!r}")

        # Single mss instance for the whole life of the object (GPU dup context)
        self._sct: Optional[mss.mss] = None
        if backend is CaptureBackend.MSS_MONITOR:
            self._init_mss()

    # ---------------------------------------------------------------------
    # Public capture API
    # ---------------------------------------------------------------------

    def grab(self) -> np.ndarray:
        """Return **BGR** frame as ``np.uint8`` array (H×W×3)."""
        if self.backend is CaptureBackend.PRINTWINDOW:
            return self._grab_printwindow()
        if self.backend is CaptureBackend.MSS_MONITOR:
            return self._grab_mss_cropped()
        raise ValueError(self.backend)

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    @staticmethod
    def match_template(
        scene: np.ndarray,
        template: np.ndarray,
        *,
        threshold: float = 0.95,
        scales: Sequence[float] | float = (1.0),
        max_results: int = 10,
        method: int = cv2.TM_CCOEFF_NORMED,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """Return up to *max_results* matches sorted by best score.

        Each item is ``((x, y), (w, h), score)`` – *x/y* are top‑left corner.
        """
        if isinstance(scales, (float, int)):
            scales = [float(scales)]

        h0, w0 = template.shape[:2]
        matches: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []

        for s in scales:
            tpl = cv2.resize(template, (0, 0), fx=s, fy=s) if s != 1.0 else template
            res = cv2.matchTemplate(scene, tpl, method)
            (ys, xs) = np.where(res >= threshold)
            for x, y in zip(xs, ys):
                score = res[y, x]
                matches.append(
                    ((int(x), int(y)), (int(w0 * s), int(h0 * s)), float(score))
                )
        # Sort & NMS
        matches.sort(key=lambda m: m[2], reverse=True)
        kept: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
        for cand in matches:
            if len(kept) >= max_results:
                break
            (cx, cy), (cw, ch), cscore = cand
            # Reject if IoU with a kept box > 0.3
            ok = True
            for (kx, ky), (kw, kh), _ in kept:
                iou = _iou((cx, cy, cw, ch), (kx, ky, kw, kh))
                if iou > 0.3:
                    ok = False
                    break
            if ok:
                kept.append(cand)
        return kept

    @staticmethod
    def detect_number(
        frame: np.ndarray,
        *,
        visualize: bool = False,
    ) -> str | None:
        def _show_steps(steps: list[tuple[str, np.ndarray]]) -> None:
            """Helper: pop up each debug image in its own window."""
            for title, img in steps:
                cv2.imshow(title, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        """
        Extract the pink power number from an OPTCG screenshot and optionally
        display each processing step.

        Parameters
        ----------
        frame : np.ndarray   BGR image captured from the game window
        visualize : bool     Pop up debug windows if True

        Returns
        -------
        str | None           Digits read (e.g. '11000') or None if no label found
        """
        debug_imgs: list[tuple[str, np.ndarray]] = [("0 – original", frame.copy())]

        # --- 1. HSV mask ------------------------------------------------------
        # TODO: Add separate check for green card buff numbers
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([147, 150, 150], dtype=np.uint8)  # H-S-V lower
        upper = np.array([163, 255, 255], dtype=np.uint8)  # H-S-V upper
        mask = cv2.inRange(hsv, lower, upper)
        debug_imgs.append(("1 – raw HSV mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))

        # --- 2. Morphology ----------------------------------------------------
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = mask
        # mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
        # mask_clean = cv2.dilate(mask_clean, k, iterations=2)
        debug_imgs.append(
            ("2 – cleaned mask", cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR))
        )

        # --- 5. OCR -----------------------------------------------------------
        config = "--psm 6 --oem 1 -c tessedit_char_whitelist=GlOIS0123456789"
        text: str = pytesseract.image_to_string(mask_clean, config=config).strip()

        # -- 6. Replace letters with numbers
        text = text.replace("O", "0")
        text = text.replace("S", "5")
        text = text.replace("G", "6")
        text = text.replace("I", "1")
        text = text.replace("l", "1")

        # ---------------------------------------------------------------------
        if visualize:
            _show_steps(debug_imgs)

        return text if text else None

    # ------------------------------------------------------------------
    # Window search helpers
    # ------------------------------------------------------------------

    def _find_main_window(self) -> Optional[int]:
        pid = _get_pid_by_name(self.exe_name)
        if pid is None:
            return None
        hwnds = _enum_hwnds_for_pid(pid)
        if not hwnds:
            return None
        # Heuristic: assume the window with largest area is the main one
        hwnds.sort(key=lambda h: _window_area(h), reverse=True)
        return hwnds[0]

    # ------------------------------------------------------------------
    # Grab strategies
    # ------------------------------------------------------------------

    def _grab_printwindow(self) -> np.ndarray:
        hwnd = self._hwnd
        if hwnd is None or not win32gui.IsWindow(hwnd):
            raise CaptureError("Window handle invalid – did the game close?")

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width, height = right - left, bottom - top
        if width == 0 or height == 0:
            raise CaptureError("Target window has zero size")

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bmp)

        # Ask Windows to render the window into our DC
        result = ctypes.windll.user32.PrintWindow(
            hwnd, save_dc.GetSafeHdc(), _PW_RENDERFULLCONTENT
        )
        if not result:
            raise CaptureError("PrintWindow failed – window may be minimised")

        # Convert raw bits → numpy arr
        bmpinfo = bmp.GetInfo()
        raw = bmp.GetBitmapBits(True)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(
            (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Release DCs to avoid handle leaks
        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        return img

    # ---- MSS‑based fallback ------------------------------------------------

    def _init_mss(self) -> None:
        if self._hwnd is None:
            raise CaptureError("Cannot set up monitor grabbing without a hwnd")
        self._sct = mss.mss()
        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        self._roi = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        }

    def _grab_mss_cropped(self) -> np.ndarray:
        if self._sct is None:
            self._init_mss()
        assert self._sct is not None  # for mypy only
        monitor = self._sct.monitors[0]  # full virtual screen
        shot = np.array(self._sct.grab(monitor))[:, :, :3]  # BGRA→BGR strip alpha
        # Crop once per call – negligible cost
        x, y, w, h = (
            self._roi["left"],
            self._roi["top"],
            self._roi["width"],
            self._roi["height"],
        )
        return shot[y : y + h, x : x + w]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _window_area(hwnd: int) -> int:
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return max(0, right - left) * max(0, bottom - top)


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area else 0.0


# ---------------------------------------------------------------------------
# Example usage (run `python -m utils.vision` for a quick sanity test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    v = OPTCGVisionHelper()  # defaults assume the sim is already running
    tpl = cv2.imread(
        str(Path(__file__).parent / "templates" / "cards" / "OP05" / "OP05-045.png")
    )
    assert tpl is not None, "Template not found for demo!"

    print("Press Ctrl‑C to exit…")
    try:
        while True:
            frame = v.grab()
            hits = OPTCGVisionHelper.detect_number(frame, visualize=True)
            print(hits)
    finally:
        cv2.destroyAllWindows()
