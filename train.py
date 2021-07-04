#!/usr/bin/env python
import random

import cv2
from bot import Evaluator
from tetris import GameState, Board, Tetrominoes


def main():
    cp = random.choice(Tetrominoes)
    cp.set_position(6, 1)
    gs = GameState(Board(), cp, None)
    frames = 0
    move_count = 0
    move_sequence = []
    while True:
        if cv2.waitKey(250) == ord("q"):
            cv2.destroyAllWindows()
            break
        gs.display()
        if gs.move_down() is False:
            break
        # if frames % 20 == 0:
        #     cp = random.choice(Tetrominoes)
        #     cp.set_position(6, 1)
        #     gs.update(gs.board, cp, None)
        frames += 1
        if gs.new_piece() and not move_sequence:
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

            best_move, time_taken = aie.best_move(
                lookahead=False, collect_final_state=False, debug=False
            )
            move_sequence = best_move.to_sequence()
            print(f"Move {move_count} found in {int(time_taken*1000)} ms.")
            print(f"\tSequence: {move_sequence}")
        if move_sequence:
            move = move_sequence.pop(0)


if __name__ == "__main__":
    main()
