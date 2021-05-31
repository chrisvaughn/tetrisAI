#!/usr/bin/env python

import argparse
import time

import cv2

from bot import Detectorist, Evaluator, execute_move
from emulator import capture, keyboard
from tetris import GameState


def main(step=False, diff_states=False, all_moves=False):
    gs = None
    move_sequence = []
    final_expected_state = None
    move_count = 0
    emulator_pid = capture.get_emulator_pid("Nestopia")
    last_time = time.time()
    for screen in capture.screenshot_generator():
        move = None
        # print("fps: {0}".format(1 / (time.time() - last_time)))
        last_time = time.time()
        if cv2.waitKey(25) == ord("q"):
            cv2.destroyAllWindows()
            break

        d = Detectorist(screen)
        # d.board.print()
        # if d.board.game_over():
        #     print("Game Over")
        #     return

        if not gs:
            print("Building GameState")
            gs = GameState(d.board, d.current_piece, d.next_piece)
        else:
            # print("Updating GameState")
            gs.update(d.board, d.current_piece, d.next_piece)

        if gs.new_piece() and not move_sequence:
            move_count += 1
            weights = {
                "holes": -0.9,
                "roughness": 0,
                "lines": 1,
                "relative_height": -0.7,
                "absolute_height": -0.8,
                "cumulative_height": 0,
            }
            aie = Evaluator(gs, weights)
            if diff_states:
                if not aie.compare_initial_to_expected(final_expected_state):
                    input("Press enter to continue.")

            best_move, time_taken = aie.best_move(
                lookahead=False, collect_final_state=diff_states, debug=all_moves
            )
            if diff_states:
                final_expected_state = best_move.final_state

            if step:
                temp_state = gs.clone()
                execute_move(temp_state, best_move.rotations, best_move.translation)
                temp_state.board.print()
            move_sequence = best_move.to_sequence()
            print(f"Move {move_count} found in {int(time_taken*1000)} ms.")
            print(f"\tScore: {best_move.score:.1f}")
            print(f"\tSequence: {move_sequence}")
            if step:
                input("Press enter to execute move.")

        if move_sequence:
            move = move_sequence.pop(0)

        keyboard.send_event(emulator_pid, move)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run tetris bot")
    parser.add_argument(
        "--step",
        action="store_true",
        help="step through each move",
    )
    parser.add_argument(
        "--all_moves",
        action="store_true",
        help="print all possible moves",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="diff expected and actual states between moves",
    )
    args = parser.parse_args()
    main(step=args.step, diff_states=args.diff, all_moves=args.all_moves)
