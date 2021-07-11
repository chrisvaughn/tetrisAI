#!/usr/bin/env python
import argparse
import random
import time

from bot import GA, Evaluator, Weights, get_pool
from tetris import Board, GameState, Tetrominoes

best_weights = Weights(
    holes= -5,
    roughness= -0.6,
    lines= 5,
    relative_height= -0.7,
    absolute_height= -0.8,
    cumulative_height= -0.5,
    well_count= 0,
)


def main(args):
    get_pool()
    ga = GA(50, 5, avg_of)
    best = ga.run()
    print("All Done")
    print(best)


def avg_of(weights, num=10):
    results = []
    for i in range(num):
        lines = evaluate(weights)
        results.append(lines)
    return sum(results) / len(results)


def evaluate(weights: Weights):
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
        if new_piece:
            new_piece = False
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
            gs.update(gs.board, gs.current_piece, None)
        else:
            moved_down = gs.move_down()
            if not moved_down:
                game_over = gs.check_game_over()
                if not game_over:
                    lines += gs.check_full_lines()
                    cp = random.choice(Tetrominoes)
                    cp.set_position(5, 1)
                    gs.update(gs.board, cp, None)
                    new_piece = True
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a tetris bot")
    parser.add_argument(
        "--seed", default=str(int(time.time() * 100000)), help="rng seed"
    )
    args = parser.parse_args()
    main(args)
