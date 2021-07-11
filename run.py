#!/usr/bin/env python

import argparse

import cv2

from bot import Detectorist, Evaluator, execute_move, get_pool
from emulator import capture, keyboard, manage
from tetris import GameState

# weights = {
#     "holes": -5,
#     "roughness": -0.6,
#     "lines": 5,
#     "relative_height": -0.7,
#     "absolute_height": -0.8,
#     "cumulative_height": -0.5,
# }

# weights = {
#     "holes": -0.5828434870040269,
#     "roughness": -0.35241321375525203,
#     "lines": 0.8371132866090609,
#     "relative_height": 0.13594466169874808,
#     "absolute_height": -0.2753119151051391,
#     "cumulative_height": -1.48472415053232,
# }

weights = {
    "holes": -1.7944608831611424,
    "roughness": -0.6830591594362199,
    "lines": 1.6440168684900818,
    "relative_height": 0.5245349681257765,
    "absolute_height": -0.6115639207004266,
    "cumulative_height": -1.663000535691957,
}


def main(all_moves=False):
    # init evaluation pool
    get_pool()
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
            print(gs.current_piece.zero_based_corner_xy)
            keyboard.send_event_off(emulator.pid, "move_down")
            drop_enabled = False
            move_count += 1
            aie.update_state(gs)
            best_move, time_taken, moves_considered = aie.best_move(debug=all_moves)

            move_sequence = best_move.to_sequence()
            print(
                f"Move {move_count}: Piece: {gs.current_piece.name}, Considered {moves_considered} moves in {int(time_taken*1000)} ms."
            )
            print(f"\tSequence: {move_sequence}")
            # print(f"\tScore: {best_move.score:.1f}")
            if best_move.lines_completed:
                lines_completed += best_move.lines_completed

        if move_sequence:
            move = move_sequence.pop(0)
            keyboard.send_event(emulator.pid, move, hold)
            drop_enabled = True
        elif drop_enabled:
            keyboard.send_event_on(emulator.pid, "move_down")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run tetris bot")
    parser.add_argument(
        "--all_moves",
        action="store_true",
        help="print all possible moves",
    )
    args = parser.parse_args()
    main(all_moves=args.all_moves)
