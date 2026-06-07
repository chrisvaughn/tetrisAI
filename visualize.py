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

HEADER_H = 46
GAP = 5
STATUS_H = 30
MENUBAR_H = 25
MAX_LINES = 230

COLOR_FILLED = (80, 200, 80)
COLOR_EMPTY = (25, 25, 25)
COLOR_BORDER = (60, 60, 60)
COLOR_TEXT = (180, 180, 180)
COLOR_STATUS = (200, 200, 100)
COLOR_BEST_LINE = (80, 220, 80)
COLOR_MEAN_LINE = (100, 210, 220)
COLOR_STD_BAND = (50, 50, 110)

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

    # Map genome id → first generation it appeared as the best performer.
    first_gen_as_best = {}
    for i, g in enumerate(hall_of_fame):
        if g.id not in first_gen_as_best:
            first_gen_as_best[g.id] = i + 1

    # Prefer hall-of-fame genomes (actually evaluated); deduplicate by id.
    seen = set()
    candidates = []
    for g in hall_of_fame + population:
        if g.id not in seen:
            seen.add(g.id)
            candidates.append(g)

    candidates.sort(key=lambda g: g.fitness, reverse=True)
    return candidates[:count], getattr(save, "generation_stats", []), first_gen_as_best


class BotGame:
    """Runs one genome's bot in a loop, restarting after each game over."""

    def __init__(self, genome, rank, gen_label, start_delay: float = 0.0):
        self.genome = genome
        self.rank = rank
        self.gen_label = gen_label
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


def render_slot(board, score, lines, rank, fitness, games, genome_id, gen_label, cell):
    bw, bh = 10 * cell, 20 * cell
    h = HEADER_H + bh
    img = np.zeros((h, bw, 3), dtype=np.uint8)
    cv2.putText(img, f"#{rank}  fit:{fitness:.0f}", (3, 13), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)
    cv2.putText(img, f"gen:{gen_label}  id:{genome_id}", (3, 27), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_STATUS, 1)
    cv2.putText(img, f"l:{lines}  g:{games}", (3, 41), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)
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
        img = render_slot(board, score, lines, bot.rank, bot.genome.fitness, games, bot.genome.id, bot.gen_label, cell)
        r0, c0 = row * slot_h, col * slot_w
        canvas[r0 : r0 + HEADER_H + bh, c0 : c0 + bw] = img

    if generation_stats:
        last = generation_stats[-1]
        status = f"Gen {last['gen']}  best:{last['best']:.1f}  mean:{last['mean']:.1f}  std:{last['std']:.1f}"
    else:
        status = "Generation 1 in progress..."
    cv2.putText(canvas, status, (5, rows * slot_h + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_STATUS, 1)
    return canvas


def make_stats_canvas(generation_stats, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(canvas, "1:Games  2:Stats  Q:Quit", (5, height - 8), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_BORDER, 1)

    if not generation_stats:
        cv2.putText(
            canvas,
            "Waiting for first generation...",
            (width // 2 - 130, height // 2),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            COLOR_TEXT,
            1,
        )
        return canvas

    ml, mr, mt, mb = 65, 30, 50, 45
    pw = width - ml - mr
    ph = height - mt - mb

    gens = [s["gen"] for s in generation_stats]
    bests = [s["best"] for s in generation_stats]
    means = [s["mean"] for s in generation_stats]
    stds = [s["std"] for s in generation_stats]

    g0, g1 = gens[0], gens[-1]
    v0, v1 = 0, max(max(bests), 10) * 1.08
    gr = max(1, g1 - g0)
    vr = max(1, v1 - v0)

    def px(g, v):
        x = ml + int((g - g0) / gr * pw)
        y = mt + ph - int((v - v0) / vr * ph)
        return x, max(mt, min(mt + ph, y))

    # Y-axis grid and labels
    v_step = 25
    for v in range(0, int(v1) + 1, v_step):
        _, y = px(g0, v)
        cv2.line(canvas, (ml, y), (ml + pw, y), (40, 40, 40), 1)
        cv2.putText(canvas, str(v), (ml - 40, y + 4), cv2.FONT_HERSHEY_PLAIN, 0.9, COLOR_TEXT, 1)

    # X-axis grid and labels — pick a round step that gives ~8 ticks
    raw_step = max(1, gr // 8)
    g_step = next((n for n in [1, 2, 5, 10, 20, 25, 50, 100, 200] if n >= raw_step), raw_step)
    first_tick = ((g0 + g_step - 1) // g_step) * g_step
    for g in range(first_tick, g1 + 1, g_step):
        x, _ = px(g, v0)
        cv2.line(canvas, (x, mt), (x, mt + ph), (40, 40, 40), 1)
        cv2.putText(canvas, str(g), (x - 8, mt + ph + 15), cv2.FONT_HERSHEY_PLAIN, 0.9, COLOR_TEXT, 1)

    # Axes border
    cv2.rectangle(canvas, (ml, mt), (ml + pw, mt + ph), (80, 80, 80), 1)

    # Std deviation shaded band
    if len(gens) > 1:
        pts_hi = [px(g, min(v1, m + s)) for g, m, s in zip(gens, means, stds)]
        pts_lo = [px(g, max(0, m - s)) for g, m, s in zip(gens, means, stds)]
        poly = np.array(pts_hi + pts_lo[::-1], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], COLOR_STD_BAND)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # Mean line
    for i in range(1, len(gens)):
        cv2.line(canvas, px(gens[i - 1], means[i - 1]), px(gens[i], means[i]), COLOR_MEAN_LINE, 1)

    # Best line (thicker)
    for i in range(1, len(gens)):
        cv2.line(canvas, px(gens[i - 1], bests[i - 1]), px(gens[i], bests[i]), COLOR_BEST_LINE, 2)

    # Legend
    lx, ly = ml + 12, mt + 16
    cv2.line(canvas, (lx, ly), (lx + 18, ly), COLOR_BEST_LINE, 2)
    cv2.putText(canvas, f"best  {bests[-1]:.1f}", (lx + 24, ly + 4), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)
    cv2.line(canvas, (lx, ly + 18), (lx + 18, ly + 18), COLOR_MEAN_LINE, 1)
    cv2.putText(canvas, f"mean  {means[-1]:.1f}", (lx + 24, ly + 22), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)
    cv2.rectangle(canvas, (lx, ly + 32), (lx + 18, ly + 42), COLOR_STD_BAND, -1)
    cv2.putText(canvas, f"±std   {stds[-1]:.1f}", (lx + 24, ly + 41), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)

    # Title
    last = generation_stats[-1]
    title = (
        f"Training Progress — Gen {last['gen']}   best:{last['best']:.1f}"
        f"   mean:{last['mean']:.1f}   std:{last['std']:.1f}"
    )
    cv2.putText(canvas, title, (ml, mt - 14), cv2.FONT_HERSHEY_PLAIN, 1.1, COLOR_STATUS, 1)

    return canvas


def start_bots(genomes, first_gen_as_best):
    bots = [
        BotGame(g, i + 1, str(first_gen_as_best.get(g.id, "pop")), start_delay=i * BOT_STAGGER_S)
        for i, g in enumerate(genomes)
    ]
    for b in bots:
        b.start()
    return bots


def stop_bots(bots):
    for b in bots:
        b.stop()


def main():
    parser = argparse.ArgumentParser(description="Visualize top training genomes as live games")
    parser.add_argument("--save-file", default="save_lines.pkl")
    parser.add_argument("--count", type=int, default=16, help="number of bots to show (default 16)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    cols, rows, cell = best_layout(args.count)
    print(f"Layout: {cols}x{rows} grid, cell={cell}px, board={10 * cell}x{20 * cell}px")

    if not os.path.isfile(args.save_file):
        print(f"Save file not found: {args.save_file}")
        print("Start training first: uv run python train.py --fitness lines")
        return

    print(f"Loading genomes (showing top {args.count})...")
    genomes, gen_stats, first_gen_as_best = load_top_genomes(args.save_file, args.count)
    print(f"Loaded {len(genomes)} genomes — staggering startup over {len(genomes) * BOT_STAGGER_S:.1f}s")

    bots = start_bots(genomes, first_gen_as_best)
    last_mtime = os.path.getmtime(args.save_file)
    frame_ms = max(1, int(1000 / args.fps))

    bw = 10 * cell
    bh = 20 * cell
    canvas_w = cols * (bw + GAP)
    canvas_h = math.ceil(len(bots) / cols) * (HEADER_H + bh + GAP) + STATUS_H

    current_tab = "games"
    print("Running. Press '1' for games view, '2' for stats graph, 'q' to quit. Auto-reloads when save file updates.")
    while True:
        if current_tab == "games":
            canvas = make_canvas(bots, cols, cell, gen_stats)
        else:
            canvas = make_stats_canvas(gen_stats, canvas_w, canvas_h)
        cv2.imshow("Tetris Training", canvas)

        key = cv2.waitKey(frame_ms) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            current_tab = "games"
        elif key == ord("2"):
            current_tab = "stats"

        try:
            mtime = os.path.getmtime(args.save_file)
        except OSError:
            continue
        if mtime != last_mtime:
            print("Save file updated — reloading genomes...")
            stop_bots(bots)
            genomes, gen_stats, first_gen_as_best = load_top_genomes(args.save_file, args.count)
            bots = start_bots(genomes, first_gen_as_best)
            last_mtime = mtime
            print(f"Loaded {len(genomes)} genomes")

    stop_bots(bots)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
