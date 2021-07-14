#!/usr/bin/env python
import argparse
import time

import cv2

from bot import GA, Evaluator, Weights, get_pool
from tetris import Board, GameState

best_weights = Weights(
    holes=-5,
    roughness=-0.6,
    lines=5,
    relative_height=-0.7,
    absolute_height=-0.8,
    cumulative_height=-0.5,
    well_count=0,
)


def main(args):
    get_pool()
    ga = GA(100, 15, avg_of)
    best = ga.run(resume=True)
    print("All Done")
    print(best)


def avg_of(weights, num=10):
    results = []
    for i in range(num):
        lines = evaluate(weights)
        results.append(lines)
    return sum(results) / len(results)


def evaluate(weights: Weights):
    gs = GameState(Board(), None, seed=time.time_ns())
    cp = gs.select_next_piece()
    gs.update(gs.board, cp)
    move_count = 0
    move_sequence = []
    game_over = False
    lines = 0
    aie = Evaluator(gs, weights)
    while not game_over:
        # if cv2.waitKey(1) == ord("q"):
        #     cv2.destroyAllWindows()
        #     break
        # gs.display()
        if gs.new_piece() and not move_sequence:
            move_count += 1
            aie.update_state(gs)
            best_move, time_taken, moves_considered = aie.best_move(debug=False)
            move_sequence = best_move.to_sequence()

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
                if not game_over:
                    lines += gs.check_full_lines()
                    cp = gs.select_next_piece()
                    gs.update(gs.board, cp)
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a tetris bot")
    parser.add_argument(
        "--seed", default=str(int(time.time() * 100000)), help="rng seed"
    )
    args = parser.parse_args()
    main(args)
