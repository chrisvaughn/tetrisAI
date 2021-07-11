#!/usr/bin/env python

import argparse
import random
import time

import cv2

from bot import Detectorist, Evaluator, Weights, get_pool
from emulator import capture, keyboard, manage
from tetris import Board, GameState, Tetrominoes

# weights = {
#     "holes": -5,
#     "roughness": -0.6,
#     "lines": 5,
#     "relative_height": -0.7,
#     "absolute_height": -0.8,
#     "cumulative_height": -0.5,
#     "well_count": 0
# }

# weights = {
#     "holes": -0.5828434870040269,
#     "roughness": -0.35241321375525203,
#     "lines": 0.8371132866090609,
#     "relative_height": 0.13594466169874808,
#     "absolute_height": -0.2753119151051391,
#     "cumulative_height": -1.48472415053232,
#     "well_count": 0
# }

weights = Weights(
    holes=-1.7944608831611424,
    roughness=-0.6830591594362199,
    lines=1.6440168684900818,
    relative_height=0.5245349681257765,
    absolute_height=-0.6115639207004266,
    cumulative_height=-1.663000535691957,
    well_count=0,
)


def main(args):
    # init evaluation pool
    get_pool()

    if args.emulator:
        run_with_emulator(args)
    else:
        run_in_memory(args)


def run_in_memory(args):
    seed = args.seed
    random.seed(seed)
    cp = random.choice(Tetrominoes)
    cp.set_position(6, 1)
    gs = GameState(Board(), cp, None)
    move_count = 0
    move_sequence = []
    game_over = False
    lines = 0
    new_piece = True
    aie = Evaluator(gs, weights)
    while not game_over:
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        gs.display()
        if new_piece:
            new_piece = False
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
            gs.update(gs.board, gs.current_piece, None)
        else:
            moved_down = gs.move_down()
            if not moved_down:
                game_over = gs.check_game_over()
                if game_over:
                    print(f"Moves: {move_count}")
                    print(f"Lines: {lines}")
                    print(f"Seed: {seed}")
                else:
                    lines += gs.check_full_lines()
                    cp = random.choice(Tetrominoes)
                    cp.set_position(5, 1)
                    gs.update(gs.board, cp, None)
                    new_piece = True


def run_with_emulator(args):

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
            gs = GameState(detector.board, detector.current_piece, detector.next_piece)
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
        "--seed", default=str(int(time.time() * 100000)), help="rng seed for non-emulator"
    )
    args = parser.parse_args()
    main(args)
