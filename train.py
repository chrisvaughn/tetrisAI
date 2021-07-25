#!/usr/bin/env python
import argparse

from bot import GA, Evaluator, Weights, get_pool
from tetris import Game


def main():
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
    game = Game()
    aie = Evaluator(game.state, weights)
    drop_enabled = False
    move_count = 0
    move_sequence = []
    game.start()
    while not game.game_over:
        if game.state.new_piece() and not move_sequence:
            drop_enabled = False
            move_count += 1
            aie.update_state(game.state)
            best_move, time_taken, moves_considered = aie.best_move(debug=False)
            move_sequence = best_move.to_sequence()

        if move_sequence:
            moves = move_sequence.pop(0)
            for move in moves:
                if move != "noop":
                    getattr(game, move)()
            game.move_seq_complete()
            drop_enabled = True
        elif drop_enabled:
            game.move_down()

    return game.lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a tetris bot")
    args = parser.parse_args()
    main()
