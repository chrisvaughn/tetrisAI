#!/usr/bin/env python
"""Live vision debugger for emulator integration.

Captures frames from the running emulator and shows:
  - Left:  raw screen with annotated detection regions
  - Right: interpreted board grid + detected piece info

Usage:
    uv run python vision_debug.py
    uv run python vision_debug.py --emulator-name FCEUX
    uv run python vision_debug.py --emulator-name Nestopia
"""

import argparse

import cv2
import numpy as np

from emulator.capture import Capture
from vision import FCEUX_OFFSETS, NESTOPIA_OFFSETS, Detectorist

CELL_PX = 20  # pixels per cell in the rendered board grid
BOARD_ROWS = 20
BOARD_COLS = 10

PANEL_H = BOARD_ROWS * CELL_PX + 60
PANEL_W = BOARD_COLS * CELL_PX + 160


def waiting_frame(msg: str) -> np.ndarray:
    canvas = np.zeros((PANEL_H, PANEL_W * 2, 3), dtype=np.uint8)
    cv2.putText(canvas, msg, (20, PANEL_H // 2), cv2.FONT_HERSHEY_PLAIN, 1.2, (100, 100, 100), 1)
    return canvas


def draw_screen_panel(screen: np.ndarray, offsets: dict) -> np.ndarray:
    colour = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)

    by0, by1, bx0, bx1 = offsets["board"]
    cv2.rectangle(colour, (bx0, by0), (bx1, by1), (0, 255, 0), 1)
    cv2.putText(colour, "board", (bx0, by0 - 3), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

    ny0, ny1, nx0, nx1 = offsets["next_piece"]
    cv2.rectangle(colour, (nx0, ny0), (nx1, ny1), (0, 255, 255), 1)
    cv2.putText(colour, "next", (nx0, ny0 - 3), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)

    return colour


def draw_board_panel(detector: Detectorist) -> np.ndarray:
    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    board = detector.board
    current = detector.current_piece
    next_p = detector.next_piece

    # Grid
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            x0 = col * CELL_PX
            y0 = row * CELL_PX + 40
            filled = board is not None and board.board[row][col] == 1
            colour = (180, 180, 180) if filled else (30, 30, 30)
            cv2.rectangle(canvas, (x0, y0), (x0 + CELL_PX - 1, y0 + CELL_PX - 1), colour, -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + CELL_PX - 1, y0 + CELL_PX - 1), (60, 60, 60), 1)

    # Current piece overlay (cyan)
    if current is not None:
        px, py = current.zero_based_corner_xy
        for cy, cx in current.cell_tuples:
            r, c = py + cy, px + cx
            if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
                x0 = c * CELL_PX
                y0 = r * CELL_PX + 40
                cv2.rectangle(canvas, (x0, y0), (x0 + CELL_PX - 1, y0 + CELL_PX - 1), (0, 220, 220), -1)
                cv2.rectangle(canvas, (x0, y0), (x0 + CELL_PX - 1, y0 + CELL_PX - 1), (60, 60, 60), 1)

    cv2.putText(canvas, "Detected Board", (2, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)

    text_x = BOARD_COLS * CELL_PX + 8
    cur_name = current.name if current else "---"
    nxt_name = next_p.name if next_p else "---"
    cur_pos = f"corner{current.zero_based_corner_xy}" if current else ""

    cv2.putText(canvas, "Current:", (text_x, 55), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 220, 220), 1)
    cv2.putText(canvas, cur_name, (text_x, 72), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)
    cv2.putText(canvas, cur_pos, (text_x, 88), cv2.FONT_HERSHEY_PLAIN, 0.8, (150, 150, 150), 1)
    cv2.putText(canvas, "Next:", (text_x, 115), cv2.FONT_HERSHEY_PLAIN, 0.85, (255, 255, 0), 1)
    cv2.putText(canvas, nxt_name, (text_x, 132), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 100), 1)

    if board is not None and board.game_over():
        cv2.putText(canvas, "GAME OVER", (2, 32), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255), 2)

    return canvas


def main(emulator_name: str):
    offsets = FCEUX_OFFSETS if "fceux" in emulator_name.lower() else NESTOPIA_OFFSETS
    capturer = Capture(emulator_name, fps=30)
    detector = Detectorist(offsets)

    print(f"Watching '{emulator_name}' window. Press Q to quit.")
    cv2.namedWindow("Vision Debug", cv2.WINDOW_NORMAL)

    while True:
        screen = capturer.get_screenshot()

        if screen is None:
            frame = waiting_frame(f"Waiting for '{emulator_name}' window...")
            cv2.imshow("Vision Debug", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            continue

        try:
            detector.update(screen)
            detector.detect_next_piece()
        except Exception as e:
            print(f"Detection error: {e}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        screen_panel = draw_screen_panel(screen, offsets)
        board_panel = draw_board_panel(detector)

        sh, sw = screen_panel.shape[:2]
        bh = board_panel.shape[0]
        scale = bh / sh
        scaled_screen = cv2.resize(screen_panel, (int(sw * scale), bh))

        cv2.imshow("Vision Debug", np.hstack([scaled_screen, board_panel]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live vision debugger")
    parser.add_argument(
        "--emulator-name",
        dest="emulator_name",
        default="fceux",
        help="window owner name to capture (default: fceux)",
    )
    args = parser.parse_args()
    main(args.emulator_name)
