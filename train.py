#!/usr/bin/env python
import argparse
import json
import random
import time

import cv2

from bot import Evaluator, get_pool
from tetris import Board, GameState, Tetrominoes

best_weights = {
    "holes": -5,
    "roughness": -0.6,
    "lines": 5,
    "relative_height": -0.7,
    "absolute_height": -0.8,
    "cumulative_height": -0.5,
}


def get_variations():
    weights = {
        "holes": -1,
        "roughness": -1,
        "lines": 1,
        "relative_height": 1,
        "absolute_height": 1,
        "cumulative_height": 1,
    }
    for i in range(100):
        weights = {k: v + i * 0.001 for k, v in weights.items()}
        yield weights


def main(args):
    # a = get_variations()
    # for i in a:
    i = best_weights
    run(args, i)


def run(args, weights):
    get_pool()
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
    while not game_over:
        if args.display:
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
            gs.display()
        if new_piece:
            new_piece = False
            move_count += 1
            aie = Evaluator(gs, weights)
            best_move, time_taken, moves_considered = aie.best_move(debug=False)
            move_sequence = best_move.to_sequence()
            if args.stats:
                print(
                    f"Move {move_count}: Piece: {gs.current_piece.name}, Considered {moves_considered} moves in {int(time_taken*1000)} ms."
                )
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
                    print(f"Moves: {move_count}")
                    print(f"Lines: {lines}")
                    print(f"Seed: {seed}")
                    print(f"Weights: {json.dumps(weights)}")
                else:
                    lines += gs.check_full_lines()
                    cp = random.choice(Tetrominoes)
                    cp.set_position(5, 1)
                    gs.update(gs.board, cp, None)
                    new_piece = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a tetris bot")
    parser.add_argument(
        "--seed", default=str(int(time.time() * 100000)), help="rng seed"
    )
    parser.add_argument(
        "--display", action="store_true", default=False, help="turn on display"
    )
    parser.add_argument(
        "--stats", action="store_true", default=False, help="print move stats display"
    )
    args = parser.parse_args()
    main(args)
