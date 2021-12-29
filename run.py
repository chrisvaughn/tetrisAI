#!/usr/bin/env python

import argparse
import os
import pickle
import time

import cv2

from bot import Detectorist, Evaluator, defined_weights, get_pool
from emulator import Emulator
from tetris import Game, GameState


def get_weights(use_save=True):
    if use_save and os.path.isfile("save.pkl"):
        with open("save.pkl", "rb") as f:
            saved = pickle.load(f)
            return saved.genomes[0].weights
    return defined_weights.best


def main(args):
    # init evaluation pool
    get_pool()
    weights = get_weights(args.saved)
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
    game = Game(seed)
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


def run_with_emulator(args, weights):
    emulator = Emulator(args.limit_speed, args.music, args.level, args.sound)
    soft_drop = args.drop
    move_sequence = []
    move_count = 0
    lines_completed = 0
    gs = GameState(args.seed)
    detector = Detectorist(5)
    aie = Evaluator(gs, weights)
    while True:
        screen = emulator.get_latest_image()
        if screen is None:
            time.sleep(0.1)
        detector.update(screen)
        if detector.board.game_over():
            print("Game Over")
            print(f"Lines Completed: {lines_completed}")
            emulator.destroy()
            return

        gs.update(detector.board, detector.current_piece, detector.next_piece)

        if gs.new_piece():
            emulator.drop_off()
            move_count += 1
            aie.update_state(gs)
            best_move, time_taken, moves_considered = aie.best_move()

            move_sequence = best_move.to_sequence()
            if args.stats:
                print(
                    f"Move {move_count}: Piece: {gs.current_piece.name}, Considered {moves_considered} moves in {int(time_taken * 1000)} ms."
                )
                print(f"\tSequence: {move_sequence}")
            if best_move.lines_completed:
                lines_completed += best_move.lines_completed

        while move_sequence:
            moves = move_sequence.pop(0)
            emulator.send_multiple_moves(moves)

        if soft_drop:
            emulator.drop_on()


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

    args = parser.parse_args()
    main(args)
