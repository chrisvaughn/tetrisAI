#!/usr/bin/env python

import argparse

import cv2

from bot import Detectorist, Evaluator, execute_move
from emulator import capture, keyboard, manage
from tetris import GameState


def main(step=False, diff_states=False, all_moves=False):
    emulator = manage.launch()

    gs = None
    move_sequence = []
    final_expected_state = None
    move_count = 0
    drop_enabled = False
    lines_completed = 0
    detector = None
    for screen in capture.screenshot_generator():
        hold = None
        if cv2.waitKey(25) == ord("q"):
            cv2.destroyAllWindows()
            break

        # cv2.imshow("Full", screen)
        if screen.shape[0] > 240:
            screen = cv2.resize(screen, (256, 240), interpolation=cv2.INTER_AREA)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Resized", screen)
        if detector is None:
            detector = Detectorist(screen, 10, 5)
        else:
            detector.update(screen)
        # detector.board.print()
        if detector.board.game_over():
            print("Game Over")
            print(f"Lines Completed: {lines_completed}")
            manage.destroy(emulator)
            return

        if not gs:
            print("Building GameState")
            gs = GameState(detector.board, detector.current_piece, detector.next_piece)
        else:
            # print("Updating GameState")
            gs.update(detector.board, detector.current_piece, detector.next_piece)

        if gs.new_piece() and not move_sequence:
            keyboard.send_event_off(emulator.pid, "move_down")
            drop_enabled = False
            move_count += 1
            weights = {
                "holes": -5,
                "roughness": -0.6,
                "lines": 5,
                "relative_height": -0.7,
                "absolute_height": -0.8,
                "cumulative_height": -0.5,
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
            print(f"\tSequence: {move_sequence}")
            # print(f"\tScore: {best_move.score:.1f}")
            if best_move.lines_completed:
                lines_completed += best_move.lines_completed

            if step:
                input("Press enter to execute move.")

        if move_sequence:
            move = move_sequence.pop(0)
            keyboard.send_event(emulator.pid, move, hold)
            drop_enabled = False
        elif drop_enabled:
            keyboard.send_event_on(emulator.pid, "move_down")


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
