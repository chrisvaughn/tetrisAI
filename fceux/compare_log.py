#!/usr/bin/env python3
"""
Compare FCEUX ground-truth log against the Python in-memory engine.

Run from the project root:
    python fceux/compare_log.py tetris_log.jsonl

Each "lock" event captures the moment $00BF (next-piece ID) changes during ARE.
At that moment $0042/$0060/$0061 still hold the just-locked piece's orientation
and pivot (x, y).  The board field is the post-line-clear board state.

Board replay (Section 2) starts from the previous event's board (empty for the
first piece), places cells using the meatfighter orientation offset table, clears
full lines, then compares against the current event's board.

NES orientation IDs (0x00-0x12, per ROM table at $8A9C):
  T: 0-3    J: 4-7    Z: 8-9    O: 10
  S: 11-12  L: 13-16  I: 17-18   (0x13 = sentinel / unassigned)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tetris.board import Board
from tetris.pieces import Tetrominoes

# ── Meatfighter orientation offset table ($8A9C) ───────────────────────────
# Key: NES orientation ID (0-18)
# Value: list of (dx, dy) cell offsets from the piece pivot stored in $0060/$0061
ORIENT_OFFSETS: dict[int, list[tuple[int, int]]] = {
    0: [(-1, 0), (0, 0), (1, 0), (0, -1)],  # T up
    1: [(0, -1), (0, 0), (1, 0), (0, 1)],  # T right
    2: [(-1, 0), (0, 0), (1, 0), (0, 1)],  # T down  (spawn)
    3: [(0, -1), (-1, 0), (0, 0), (0, 1)],  # T left
    4: [(0, -1), (0, 0), (-1, 1), (0, 1)],  # J left
    5: [(-1, -1), (-1, 0), (0, 0), (1, 0)],  # J up
    6: [(0, -1), (1, -1), (0, 0), (0, 1)],  # J right
    7: [(-1, 0), (0, 0), (1, 0), (1, 1)],  # J down  (spawn)
    8: [(-1, 0), (0, 0), (0, 1), (1, 1)],  # Z horizontal (spawn)
    9: [(1, -1), (0, 0), (1, 0), (0, 1)],  # Z vertical
    10: [(-1, 0), (0, 0), (-1, 1), (0, 1)],  # O        (spawn, only orientation)
    11: [(0, 0), (1, 0), (-1, 1), (0, 1)],  # S horizontal (spawn)
    12: [(0, -1), (0, 0), (1, 0), (1, 1)],  # S vertical
    13: [(0, -1), (0, 0), (0, 1), (1, 1)],  # L right
    14: [(-1, 0), (0, 0), (1, 0), (-1, 1)],  # L down  (spawn)
    15: [(-1, -1), (0, -1), (0, 0), (0, 1)],  # L left
    16: [(1, -1), (-1, 0), (0, 0), (1, 0)],  # L up
    17: [(0, -2), (0, -1), (0, 0), (0, 1)],  # I vertical
    18: [(-2, 0), (-1, 0), (0, 0), (1, 0)],  # I horizontal (spawn)
    # 19 (0x13) = "second vertical" I-piece state (same offsets as ID 17).
    # NES uses this sentinel ID after rotating I through horizontal→vertical→horizontal.
    # Board capture is unreliable for this ID: $00BF sometimes fires before the
    # line-clear row-shift animation completes, leaving $0400 in a mid-clear state.
    19: [(0, -2), (0, -1), (0, 0), (0, 1)],  # I vertical (alt, same as 17)
}

# NES orientation ID → NES piece type (0-6)
ORIENT_TO_NES_TYPE = {
    **{k: 0 for k in range(0, 4)},  # T
    **{k: 1 for k in range(4, 8)},  # J
    **{k: 2 for k in range(8, 10)},  # Z
    10: 3,  # O
    **{k: 4 for k in range(11, 13)},  # S
    **{k: 5 for k in range(13, 17)},  # L
    **{k: 6 for k in range(17, 19)},  # I
}

# NES piece type → Python Tetrominoes index
NES_TYPE_TO_PYTHON = {0: 5, 1: 2, 2: 6, 3: 3, 4: 4, 5: 1, 6: 0}
# T→t  J→j  Z→z  O→o  S→s  L→l  I→i

PIECE_NAMES = {v: Tetrominoes[v].name for v in range(len(Tetrominoes))}


# ── Helpers ────────────────────────────────────────────────────────────────


def board_from_str(s: str) -> np.ndarray:
    return np.array([int(c) for c in s], dtype=int).reshape(Board.rows, Board.columns)


def place_and_clear(board: np.ndarray, orient_id: int, px: int, py: int) -> tuple[np.ndarray, int]:
    """Place 4 cells using meatfighter offsets, clear full lines. Returns (new_board, lines)."""
    offsets = ORIENT_OFFSETS.get(orient_id)
    if offsets is None:
        raise ValueError(f"Unknown orientation ID {orient_id}")

    result = board.copy()
    for dx, dy in offsets:
        r, c = py + dy, px + dx
        if not (0 <= r < Board.rows and 0 <= c < Board.columns):
            raise ValueError(f"Cell ({r},{c}) out of bounds for orient_id={orient_id} pivot=({px},{py})")
        if result[r, c] != 0:
            raise ValueError(f"Cell ({r},{c}) already occupied for orient_id={orient_id} pivot=({px},{py})")
        result[r, c] = 1

    full = np.all(result != 0, axis=1)
    n_cleared = int(full.sum())
    if n_cleared:
        cleared = result[~full]
        empty = np.zeros((n_cleared, Board.columns), dtype=int)
        result = np.vstack([empty, cleared])
    return result, n_cleared


def print_board_diff(python_board: np.ndarray, nes_board: np.ndarray):
    print("    col: 0123456789    col: 0123456789")
    for r in range(Board.rows):
        py_row = "".join(str(v) for v in python_board[r])
        nes_row = "".join(str(v) for v in nes_board[r])
        marker = " <--" if py_row != nes_row else ""
        print(f"    r{r:02d}: {py_row}    r{r:02d}: {nes_row}{marker}")


def parse_log(path: str) -> list[dict]:
    events = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["event"] == "lock":
                events.append(entry)
    return events


# ── Section 1: Piece sequence ───────────────────────────────────────────────


def analyze_piece_sequence(events: list[dict]):
    print("=" * 70)
    print("SECTION 1: Piece sequence")
    print("=" * 70)
    print(f"Total pieces logged: {len(events)}")
    print()

    print(
        f"{'i':>4}  {'oid':>3}  {'oa':>3}  {'ob':>3}  {'type':>5}  "
        f"{'name':>5}  {'x':>3}  {'y':>3}  {'prng':>6}  {'lines':>5}"
    )
    print("-" * 65)
    for e in events[:30]:
        oid = e["piece_id"]
        oa = e.get("orient_a", oid)
        ob = e.get("orient_b", oid)
        nes_type = ORIENT_TO_NES_TYPE.get(oid, -1)
        py_idx = NES_TYPE_TO_PYTHON.get(nes_type, -1)
        name = Tetrominoes[py_idx].name if py_idx >= 0 else "?"
        flag = " !" if oa != ob else ""
        print(
            f"  {e['i']:>4}  {oid:>3}  {oa:>3}  {ob:>3}  {nes_type:>5}  {name:>5}  "
            f"{e['x']:>3}  {e['y']:>3}  0x{e['prng']:04X}  {e['lines']:>5}{flag}"
        )
    print()


# ── Section 2: Board replay ─────────────────────────────────────────────────


def replay_boards(events: list[dict]):
    print()
    print("=" * 70)
    print("SECTION 2: Board replay (using meatfighter orientation offsets)")
    print("=" * 70)
    print()

    matches = 0
    mismatches = 0
    skipped = 0

    for i, ev in enumerate(events):
        if i == 0:
            prev_board = np.zeros((Board.rows, Board.columns), dtype=int)
        else:
            prev_board = board_from_str(events[i - 1]["board"])

        oid = ev["piece_id"]
        px, py = ev["x"], ev["y"]
        expected = board_from_str(ev["board"])

        if oid not in ORIENT_OFFSETS:
            print(f"  Piece {ev['i']:3d} (orient={oid}): SKIP — sentinel/unknown orientation ID")
            skipped += 1
            continue

        try:
            got, n_cleared = place_and_clear(prev_board, oid, px, py)
        except ValueError as e:
            piece_name = PIECE_NAMES.get(NES_TYPE_TO_PYTHON.get(ORIENT_TO_NES_TYPE.get(oid, -1), -1), "?")
            print(f"  Piece {ev['i']:3d} (orient={oid} {piece_name} x={px} y={py}): ERROR — {e}")
            mismatches += 1
            continue

        if np.array_equal(got, expected):
            matches += 1
            if i < 15:
                nes_type = ORIENT_TO_NES_TYPE.get(oid, -1)
                name = PIECE_NAMES.get(NES_TYPE_TO_PYTHON.get(nes_type, -1), "?")
                print(f"  Piece {ev['i']:3d} (orient={oid:2d} {name:1s}  x={px} y={py}  cleared={n_cleared}): MATCH")
        else:
            mismatches += 1
            diff_cells = int((got != expected).sum())
            nes_type = ORIENT_TO_NES_TYPE.get(oid, -1)
            name = PIECE_NAMES.get(NES_TYPE_TO_PYTHON.get(nes_type, -1), "?")
            print(
                f"  Piece {ev['i']:3d} (orient={oid:2d} {name:1s}  x={px} y={py}  "
                f"cleared={n_cleared}): MISMATCH — {diff_cells} cells differ"
            )
            if mismatches <= 3:
                print("  Python result (left) vs NES expected (right):")
                print_board_diff(got, expected)

    print()
    print(f"Results: {matches} match, {mismatches} mismatch, {skipped} skipped (sentinel) out of {len(events)} pieces")
    if mismatches > 0:
        print()
        print("Diagnosis guide:")
        print("  Offset error         -> NES x/y coordinate different from meatfighter pivot")
        print("  All cells off by 1   -> ADDR_PIECE_X/Y ($0060/$0061) vs ($0040/$0041) mismatch")
        print("  Wrong shape          -> fix ORIENT_TO_NES_TYPE or NES_TYPE_TO_PYTHON")
        print("  Sentinel pieces      -> I-piece board-capture timing issue (needs investigation)")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} tetris_log.jsonl")
        sys.exit(1)

    log_path = sys.argv[1]
    if not Path(log_path).exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    events = parse_log(log_path)
    print(f"Loaded {len(events)} lock events\n")

    if not events:
        print("No lock events found — check that the Lua script loaded correctly.")
        sys.exit(1)

    analyze_piece_sequence(events)
    replay_boards(events)


if __name__ == "__main__":
    main()
