#!/usr/bin/env python

import argparse
import os
import pickle
import time
from collections import Counter

import cv2

from bot import Detectorist, Evaluator, defined_weights, get_pool
from tetris import Game, GameState, Tetrominoes


def get_weights(mode, use_save=True):
    if use_save and os.path.isfile(f"save_{mode}.pkl"):
        with open(f"save_{mode}.pkl", "rb") as f:
            saved = pickle.load(f)
            return saved.genomes[0].weights
    print(f"Getting weights for mode: {mode}")
    return defined_weights.by_mode[mode]


def print_final_stats(lines: int, piece_stats: Counter, combos: Counter):
    print(f"Lines Completed: {lines}")
    print(f"Piece Stats: Total {sum(piece_stats.values())}")
    for piece in Tetrominoes:
        print(f"\t{piece.name.upper()}: {piece_stats[piece.name]}")
    print("Line Combos:")
    for i in range(1, 5):
        print(f"\t{i}: {combos[i]}")


def main(args):
    # init evaluation pool
    get_pool()
    weights = get_weights(args.mode, args.saved)
    print(f"{weights}")
    if args.emulator:
        run_with_emulator(args, weights)
    else:
        run_in_memory(args, weights)


def run_in_memory(args, weights):
    seed = args.seed
    soft_drop = args.drop
    move_count = 0
    drop_enabled = False
    move_sequence = []
    game = Game(seed, args.level)
    game.display()
    aie = Evaluator(game.state, weights)
    game.start()
    while not game.game_over:
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        game.display()
        if game.state.new_piece() and not move_sequence:
            drop_enabled = False
            move_count += 1
            aie.update_state(game.state)
            best_move, time_taken, moves_considered = aie.best_move(debug=False)
            move_sequence = best_move.to_sequence()
            if args.stats:
                print(
                    f"Move {move_count}: Piece: {game.state.current_piece.name}, Considered {moves_considered} moves in {int(time_taken * 1000)} ms."
                )
                print(f"\tSequence: {move_sequence}")
        if move_sequence:
            moves = move_sequence.pop(0)
            for move in moves:
                if move != "noop":
                    getattr(game, move)()
            game.move_seq_complete()
            drop_enabled = True
        elif drop_enabled and soft_drop:
            game.move_down()

    print("Game Over")
    print_final_stats(game.lines, game.piece_stats, game.line_combos)


def run_with_emulator(args, weights):
    from emulator import Emulator

    emulator = Emulator(args.limit_speed, args.music, args.level, args.sound)
    soft_drop = args.drop
    move_sequence = []
    move_count = 0
    lines_completed = 0
    line_combos = Counter()
    piece_stats = Counter()
    gs = GameState(args.seed)
    detector = Detectorist(100)
    aie = Evaluator(gs, weights)
    while True:
        screen = emulator.get_latest_image()
        if screen is None:
            time.sleep(0.01)
            continue
        detector.update(screen)
        if detector.board.game_over():
            break

        gs.update(detector.board, detector.current_piece, detector.next_piece)

        if gs.new_piece():
            emulator.drop_off()
            move_count += 1
            aie.update_state(gs)
            best_move, time_taken, moves_considered = aie.best_move()
            piece_stats[gs.current_piece.name] += 1
            move_sequence = best_move.to_sequence()
            if args.stats:
                print(
                    f"Move {move_count}: Piece: {gs.current_piece.name}, Considered {moves_considered} moves in {int(time_taken * 1000)} ms."
                )
                print(f"\tSequence: {move_sequence}")
            if best_move.lines_completed:
                lines = best_move.lines_completed
                line_combos[lines] += 1
                lines_completed += lines

        while move_sequence:
            moves = move_sequence.pop(0)
            emulator.send_multiple_moves(moves)

        if soft_drop:
            emulator.drop_on()

    print("Game Over")
    print_final_stats(lines_completed, piece_stats, line_combos)
    emulator.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run tetris bot")
    parser.add_argument(
        "--emulator", action="store_true", default=False, help="run with emulator"
    )
    parser.add_argument(
        "--stats", action="store_true", default=False, help="print move stats display"
    )
    parser.add_argument(
        "--seed",
        default=int(time.time() * 100000),
        help="rng seed for non-emulator",
    )
    parser.add_argument(
        "--saved",
        action="store_true",
        default=False,
        help="use weights from save file if present",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        default=False,
        help="soft drop pieces",
    )
    parser.add_argument(
        "--limit-speed",
        dest="limit_speed",
        action="store_true",
        default=False,
        help="applies cheat to emulator to limit speed to level 19",
    )
    parser.add_argument(
        "--sound", action="store_true", default=False, help="enable all sounds"
    )
    parser.add_argument(
        "--music", action="store_true", default=False, help="play music"
    )
    parser.add_argument("--level", default=19, type=int, help="level to start at")
    parser.add_argument(
        "--mode",
        default="lines",
        choices=["lines", "score"],
        help="play for lines or for score",
    )

    args = parser.parse_args()
    main(args)
