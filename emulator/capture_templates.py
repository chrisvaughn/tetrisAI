#!/usr/bin/env python
"""One-shot helper for capturing FCEUX screen templates.

Run this while FCEUX is open, navigate to the screen you want to capture,
then press Enter.  The script saves the current frame as the named template
and exits.

Usage:
    uv run python emulator/capture_templates.py game_type_select
    uv run python emulator/capture_templates.py level_select
    uv run python emulator/capture_templates.py music_select
"""

import os
import sys

import cv2

from emulator.capture import Capture

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
EMULATOR_NAME = "fceux"


def main():
    if len(sys.argv) < 2:
        print("Usage: capture_templates.py <template_name>")
        sys.exit(1)

    name = sys.argv[1]
    out_path = os.path.join(TEMPLATES_DIR, f"{name}.png")

    capturer = Capture(EMULATOR_NAME, fps=30)
    print(f"Watching '{EMULATOR_NAME}' window.")
    print(f"Navigate to the '{name}' screen, then press Enter to capture.")

    # Show a live preview in a cv2 window so the user can see what will be captured.
    print("Press  S or SPACE  inside the preview window to save.  Q to quit without saving.")
    cv2.namedWindow("Capture Preview", cv2.WINDOW_NORMAL)
    screen = None
    saved = False
    while True:
        frame = capturer.get_screenshot()
        if frame is not None:
            screen = frame
            cv2.imshow("Capture Preview", screen)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("s"), ord(" ")):
            saved = True
            break
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if not saved or screen is None:
        print("Cancelled — no template saved.")
        sys.exit(0)

    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    cv2.imwrite(out_path, screen)
    print(f"Saved template → {out_path}  ({screen.shape[1]}×{screen.shape[0]})")


if __name__ == "__main__":
    main()
