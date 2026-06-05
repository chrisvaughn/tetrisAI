#!/usr/bin/env python
"""
Show the top N genomes from a training save file as live simultaneous games.

Usage:
    uv run python visualize.py                      # 9 bots (3x3), safe alongside training
    uv run python visualize.py --count 25           # 5x5 grid (pause training first)
    uv run python visualize.py --save-file save_score.pkl

Press 'q' to quit. The display auto-reloads when the save file changes.
"""
import argparse
import math
import os
import pickle
import threading
import time

import cv2
import numpy as np

from bot import WeightedBot
from tetris import Game

HEADER_H = 32
GAP = 5
STATUS_H = 30
MENUBAR_H = 25
MAX_LINES = 230

COLOR_FILLED = (80, 200, 80)
COLOR_EMPTY = (25, 25, 25)
COLOR_BORDER = (60, 60, 60)
COLOR_TEXT = (180, 180, 180)
COLOR_STATUS = (200, 200, 100)

BOT_STAGGER_S = 0.2  # stagger startup so bots don't all evaluate their first move at once


def screen_size():
    try:
        import Quartz
        b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(b.size.width), int(b.size.height)
    except Exception:
        return 1920, 1080


def best_layout(count):
    """Find the (cols, rows, cell_px) that maximises cell size on the current screen."""
    sw, sh = screen_size()
    usable_h = sh - MENUBAR_H - STATUS_H
    best_cell, best_cols = 0, 1
    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        cell_h = (usable_h - rows * (HEADER_H + GAP)) // (rows * 20)
        cell_w = (sw - cols * GAP) // (cols * 10)
        cell = min(cell_h, cell_w)
        if cell > best_cell:
            best_cell, best_cols = cell, cols
    return best_cols, math.ceil(count / best_cols), best_cell


def load_top_genomes(save_file, count):
    # Save files are written by the training process — trusted local data only.
    with open(save_file, "rb") as f:
        save = pickle.load(f)

    hall_of_fame = list(save.best_for_each_generation or [])
    population = list(save.genomes or [])

    # Prefer hall-of-fame genomes (actually evaluated); deduplicate by id.
    seen = set()
    candidates = []
    for g in hall_of_fame + population:
        if g.id not in seen:
            seen.add(g.id)
            candidates.append(g)

    candidates.sort(key=lambda g: g.fitness, reverse=True)
    return candidates[:count], getattr(save, "generation_stats", [])


class BotGame:
    """Runs one genome's bot in a loop, restarting after each game over."""

    def __init__(self, genome, rank, start_delay: float = 0.0):
        self.genome = genome
        self.rank = rank
        self._start_delay = start_delay
        self._lock = threading.Lock()
        self._board = np.zeros((20, 10), dtype=int)
        self._score = 0
        self._lines = 0
        self._games_played = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True

    def snapshot(self):
        with self._lock:
            return self._board.copy(), self._score, self._lines, self._games_played

    def _loop(self):
        if self._start_delay > 0:
            time.sleep(self._start_delay)
        while not self._stop:
            self._play_game()
            if not self._stop:
                time.sleep(0.5)

    def _play_game(self):
        game = Game(level=19)
        bot = WeightedBot(self.genome.weights, parallel=False, scoring="v2")
        game.start()
        move_seq = []
        while not game.game_over and not self._stop:
            if game.lines >= MAX_LINES:
                break
            if game.state.new_piece() and not move_seq:
                bot.update_state(game.state)
                try:
                    best, _, _ = bot.get_best_move(debug=False)
                    move_seq = best.to_sequence()
                except Exception:
                    pass
            if move_seq:
                moves = move_seq.pop(0)
                for m in moves:
                    if m != "noop":
                        getattr(game, m)()
                game.move_seq_complete()
            else:
                # Yield to other threads while waiting for the next new piece.
                time.sleep(0.005)
            with self._lock:
                self._board = game.state.board.board.copy()
                self._score = game.score
                self._lines = game.lines
        with self._lock:
            self._games_played += 1


def render_slot(board, score, lines, rank, fitness, games, cell):
    bw, bh = 10 * cell, 20 * cell
    h = HEADER_H + bh
    img = np.zeros((h, bw, 3), dtype=np.uint8)
    cv2.putText(img, f"#{rank}  fit:{fitness:.0f}", (3, 13), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)
    cv2.putText(img, f"s:{score}  l:{lines}  g:{games}", (3, 27), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)
    for r in range(20):
        for c in range(10):
            x0, y0 = c * cell, HEADER_H + r * cell
            color = COLOR_FILLED if board[r, c] else COLOR_EMPTY
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + cell - 1, y0 + cell - 1), color, -1)
    cv2.rectangle(img, (0, HEADER_H), (bw - 1, h - 1), COLOR_BORDER, 1)
    return img


def make_canvas(bots, cols, cell, generation_stats):
    bw, bh = 10 * cell, 20 * cell
    rows = math.ceil(len(bots) / cols)
    slot_w = bw + GAP
    slot_h = HEADER_H + bh + GAP
    canvas = np.zeros((rows * slot_h + STATUS_H, cols * slot_w, 3), dtype=np.uint8)

    for i, bot in enumerate(bots):
        board, score, lines, games = bot.snapshot()
        row, col = divmod(i, cols)
        img = render_slot(board, score, lines, bot.rank, bot.genome.fitness, games, cell)
        r0, c0 = row * slot_h, col * slot_w
        canvas[r0:r0 + HEADER_H + bh, c0:c0 + bw] = img

    if generation_stats:
        last = generation_stats[-1]
        status = f"Gen {last['gen']}  best:{last['best']:.1f}  mean:{last['mean']:.1f}  std:{last['std']:.1f}"
    else:
        status = "Generation 1 in progress..."
    cv2.putText(canvas, status, (5, rows * slot_h + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_STATUS, 1)
    return canvas


def start_bots(genomes):
    bots = [BotGame(g, i + 1, start_delay=i * BOT_STAGGER_S) for i, g in enumerate(genomes)]
    for b in bots:
        b.start()
    return bots


def stop_bots(bots):
    for b in bots:
        b.stop()


def main():
    parser = argparse.ArgumentParser(description="Visualize top training genomes as live games")
    parser.add_argument("--save-file", default="save_lines.pkl")
    parser.add_argument("--count", type=int, default=16,
                        help="number of bots to show (default 16)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    cols, rows, cell = best_layout(args.count)
    print(f"Layout: {cols}x{rows} grid, cell={cell}px, board={10*cell}x{20*cell}px")

    if not os.path.isfile(args.save_file):
        print(f"Save file not found: {args.save_file}")
        print("Start training first: uv run python train.py --fitness lines")
        return

    print(f"Loading genomes (showing top {args.count})...")
    genomes, gen_stats = load_top_genomes(args.save_file, args.count)
    print(f"Loaded {len(genomes)} genomes — staggering startup over {len(genomes) * BOT_STAGGER_S:.1f}s")

    bots = start_bots(genomes)
    last_mtime = os.path.getmtime(args.save_file)
    frame_ms = max(1, int(1000 / args.fps))

    print("Running. Press 'q' to quit. Display auto-reloads when save file updates.")
    while True:
        canvas = make_canvas(bots, cols, cell, gen_stats)
        cv2.imshow("Tetris Training", canvas)

        key = cv2.waitKey(frame_ms) & 0xFF
        if key == ord("q"):
            break

        try:
            mtime = os.path.getmtime(args.save_file)
        except OSError:
            continue
        if mtime != last_mtime:
            print("Save file updated — reloading genomes...")
            stop_bots(bots)
            genomes, gen_stats = load_top_genomes(args.save_file, args.count)
            bots = start_bots(genomes)
            last_mtime = mtime
            print(f"Loaded {len(genomes)} genomes")

    stop_bots(bots)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
