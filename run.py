#!/usr/bin/env python

import argparse
import os
import pickle
import time

import cv2

from bot import Detectorist, Evaluator, defined_weights, get_pool
from emulator import capture, keyboard, manage
from tetris import Board, GameState


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
    gs = GameState(Board(), None, None, seed)
    cp = gs.select_next_piece()
    gs.update(gs.board, cp, None)
    move_count = 0
    move_sequence = []
    game_over = False
    lines = 0
    aie = Evaluator(gs, weights)
    while not game_over:
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        gs.display()
        if gs.new_piece() and not move_sequence:
            move_count += 1
            aie.update_state(gs)
            best_move, time_taken, moves_considered = aie.best_move(debug=False)
            move_sequence = best_move.to_sequence()
            if args.stats:
                print(
                    f"Move {move_count}: Piece: {gs.current_piece.name}, Considered {moves_considered} moves in {int(time_taken * 1000)} ms."
                )
                print(f"\tSequence: {move_sequence}")
        if move_sequence:
            moves = move_sequence.pop(0)
            for move in moves:
                if move != "noop":
                    getattr(gs, move)()
            gs.move_down()
            gs.update(gs.board, gs.current_piece)
        else:
            moved_down = gs.move_down()
            gs.update(gs.board, cp)
            if not moved_down:
                game_over = gs.check_game_over()
                if game_over:
                    print(f"Moves: {move_count}")
                    print(f"Lines: {lines}")
                    print(f"Seed: {seed}")
                else:
                    lines += gs.check_full_lines()
                    cp = gs.select_next_piece()
                    gs.update(gs.board, cp)


def run_with_emulator(args, weights):

    emulator = manage.launch()

    gs = None
    move_sequence = []
    move_count = 0
    drop_enabled = False
    lines_completed = 0
    detector = None
    aie = Evaluator(gs, weights)
    for screen in capture.screenshot_generator():
        hold = None
        if screen.shape[0] > 240:
            screen = cv2.resize(screen, (256, 240), interpolation=cv2.INTER_AREA)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        if detector is None:
            detector = Detectorist(screen, 10, 5)
        else:
            detector.update(screen)
        if detector.board.game_over():
            print("Game Over")
            print(f"Lines Completed: {lines_completed}")
            manage.destroy(emulator)
            return

        if not gs:
            print("Building GameState")
            gs = GameState(
                detector.board, detector.current_piece, detector.next_piece, 0
            )
        else:
            gs.update(detector.board, detector.current_piece, detector.next_piece)

        if gs.new_piece() and not move_sequence:
            keyboard.send_event_off(emulator.pid, "move_down")
            drop_enabled = False
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

        if move_sequence:
            move = move_sequence.pop(0)
            keyboard.send_events(emulator.pid, move, hold)
            drop_enabled = True
        elif drop_enabled:
            keyboard.send_event_on(emulator.pid, "move_down")


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
    args = parser.parse_args()
    main(args)
