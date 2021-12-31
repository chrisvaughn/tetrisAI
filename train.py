#!/usr/bin/env python
import argparse

from bot import GA, Evaluator, Weights, get_pool
from tetris import Game

run_evaluator_in_parallel = True


def main(args):
    global run_evaluator_in_parallel
    if args.no_parallel:
        run_evaluator_in_parallel = False
    get_pool()
    fitness_methods = {
        "score": avg_of_scores,
        "lines": avg_of_lines,
    }
    filename = f"save_{args.fitness_method}.pkl"
    ga = GA(100, 15, fitness_methods[args.fitness_method], filename)
    best = ga.run(resume=True)
    print("All Done")
    print(best)


def avg_of_scores(weights, num=10):
    results = []
    for i in range(num):
        lines, score = evaluate(weights, run_evaluator_in_parallel)
        results.append(score)
    return sum(results) / len(results)


def avg_of_lines(weights, num=10):
    results = []
    for i in range(num):
        lines, score = evaluate(weights, run_evaluator_in_parallel)
        results.append(lines)
    return sum(results) / len(results)


def evaluate(weights: Weights, parallel: bool = True):
    game = Game(level=19)
    aie = Evaluator(game.state, weights, parallel)
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

    return game.lines, game.score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a tetris bot")
    parser.add_argument(
        "--no-parallel",
        dest="no_parallel",
        action="store_true",
        default=False,
        help="do not run evaluator in parallel",
    )
    parser.add_argument(
        "--fitness",
        dest="fitness_method",
        default="lines",
        choices=["lines", "score"],
        help="fitness method",
    )
    args = parser.parse_args()
    main(args)
