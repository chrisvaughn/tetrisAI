#!/usr/bin/env python
"""
Show the top 25 genomes from a training save file as live simultaneous games.

Usage:
    uv run python visualize.py
    uv run python visualize.py --save-file save_score.pkl
    uv run python visualize.py --fps 15

Press 'q' to quit. The display auto-reloads when the save file changes.
"""
import argparse
import os
import pickle
import threading
import time

import cv2
import numpy as np

from bot import WeightedBot
from bot.weighted_bot.evolution import SaveState
from tetris import Game

CELL = 8
BOARD_W = 10 * CELL   # 80px
BOARD_H = 20 * CELL   # 160px
HEADER_H = 22
GAP = 5
COLS = 5
ROWS = 5
N = COLS * ROWS        # 25 bots
MAX_LINES = 230

COLOR_FILLED = (80, 200, 80)
COLOR_EMPTY = (25, 25, 25)
COLOR_GRID = (50, 50, 50)
COLOR_TEXT = (180, 180, 180)
COLOR_BORDER = (60, 60, 60)


def load_top_genomes(save_file):
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
    return candidates[:N], getattr(save, "generation_stats", [])


class BotGame:
    """Runs one genome's bot in a loop, restarting after each game."""

    def __init__(self, genome, rank):
        self.genome = genome
        self.rank = rank
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
            with self._lock:
                self._board = game.state.board.board.copy()
                self._score = game.score
                self._lines = game.lines
        with self._lock:
            self._games_played += 1


def render_slot(board, score, lines, rank, fitness, games):
    h = HEADER_H + BOARD_H
    img = np.zeros((h, BOARD_W, 3), dtype=np.uint8)

    # Header
    gen_label = f"#{rank}  fit:{fitness:.0f}"
    game_label = f"s:{score} l:{lines} g:{games}"
    cv2.putText(img, gen_label, (2, 9), cv2.FONT_HERSHEY_PLAIN, 0.75, COLOR_TEXT, 1)
    cv2.putText(img, game_label, (2, 19), cv2.FONT_HERSHEY_PLAIN, 0.75, COLOR_TEXT, 1)

    # Board cells
    for r in range(20):
        for c in range(10):
            x0 = c * CELL
            y0 = HEADER_H + r * CELL
            color = COLOR_FILLED if board[r, c] else COLOR_EMPTY
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + CELL - 1, y0 + CELL - 1), color, -1)

    # Outer border
    cv2.rectangle(img, (0, HEADER_H), (BOARD_W - 1, h - 1), COLOR_BORDER, 1)
    return img


def make_canvas(bots, generation_stats):
    slot_w = BOARD_W + GAP
    slot_h = HEADER_H + BOARD_H + GAP
    status_h = 30
    canvas = np.zeros((ROWS * slot_h + status_h, COLS * slot_w, 3), dtype=np.uint8)

    for i, bot in enumerate(bots):
        board, score, lines, games = bot.snapshot()
        row, col = divmod(i, COLS)
        img = render_slot(board, score, lines, bot.rank, bot.genome.fitness, games)
        r0 = row * slot_h
        c0 = col * slot_w
        canvas[r0:r0 + HEADER_H + BOARD_H, c0:c0 + BOARD_W] = img

    # Status bar
    if generation_stats:
        last = generation_stats[-1]
        status = (f"Gen {last['gen']}  "
                  f"best:{last['best']:.1f}  "
                  f"mean:{last['mean']:.1f}  "
                  f"std:{last['std']:.1f}")
    else:
        status = "Generation 1 in progress..."
    cv2.putText(canvas, status, (5, ROWS * slot_h + 20),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 100), 1)

    return canvas


def start_bots(genomes):
    bots = [BotGame(g, i + 1) for i, g in enumerate(genomes)]
    for b in bots:
        b.start()
    return bots


def stop_bots(bots):
    for b in bots:
        b.stop()


def main():
    parser = argparse.ArgumentParser(description="Visualize top 25 training genomes")
    parser.add_argument("--save-file", default="save_lines.pkl")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if not os.path.isfile(args.save_file):
        print(f"Save file not found: {args.save_file}")
        print("Start training first: uv run python train.py --fitness lines")
        return

    print("Loading genomes...")
    genomes, gen_stats = load_top_genomes(args.save_file)
    print(f"Loaded {len(genomes)} genomes — starting {len(genomes)} games")

    bots = start_bots(genomes)
    last_mtime = os.path.getmtime(args.save_file)
    frame_ms = max(1, int(1000 / args.fps))

    print("Running. Press 'q' to quit. Display auto-reloads when save file updates.")
    while True:
        canvas = make_canvas(bots, gen_stats)
        cv2.imshow("Tetris Training", canvas)

        key = cv2.waitKey(frame_ms) & 0xFF
        if key == ord("q"):
            break

        # Auto-reload when save file changes
        try:
            mtime = os.path.getmtime(args.save_file)
        except OSError:
            continue
        if mtime != last_mtime:
            print("Save file updated — reloading genomes...")
            stop_bots(bots)
            genomes, gen_stats = load_top_genomes(args.save_file)
            bots = start_bots(genomes)
            last_mtime = mtime
            print(f"Loaded {len(genomes)} genomes")

    stop_bots(bots)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
