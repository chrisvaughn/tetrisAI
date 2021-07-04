#!/usr/bin/env python
import random

import cv2

from bot import Evaluator
from tetris import Board, GameState, Tetrominoes

random.seed(1)


def main():
    cp = random.choice(Tetrominoes)
    cp.set_position(6, 1)
    gs = GameState(Board(), cp, None)
    move_count = 0
    move_sequence = []
    game_over = False
    lines = 0
    new_piece = True
    while not game_over:
        if cv2.waitKey(100) == ord("q"):
            cv2.destroyAllWindows()
            break
        gs.display()
        if new_piece:
            new_piece = False
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
            best_move, time_taken = aie.best_move(collect_final_state=False, debug=True)
            move_sequence = best_move.to_sequence()
            print(f"Move {move_count} found in {int(time_taken*1000)} ms.")
            print(f"\tSequence: {move_sequence}")
        if move_sequence:
            move = move_sequence.pop(0)
            if move != "noop":
                getattr(gs, move)()
            gs.move_down()
            gs.update(gs.board, gs.current_piece, None)
        else:
            moved_down = gs.move_down()
            if not moved_down:
                game_over = gs.check_game_over()
                if game_over:
                    print("Game Over")
                    print(f"Total Lines {lines}")
                else:
                    lines += gs.check_full_lines()
                    cp = random.choice(Tetrominoes)
                    print(f"New Piece {cp.name}")
                    cp.set_position(5, 1)
                    gs.update(gs.board, cp, None)
                    new_piece = True


if __name__ == "__main__":
    main()
