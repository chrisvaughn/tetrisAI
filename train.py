#!/usr/bin/env python
import argparse
import statistics

from bot import GA, Evaluator, Weights, get_pool
from tetris import Game

run_evaluator_in_parallel = True
training_seeds = [
    123456789,
    0,
    1,
    10410112132101108105,
    110111114105,
    3364115110111114105,
    546910785476980,
    14578341609431,
    112117112112121,
    2356441353364364,
]


def main(args):
    global run_evaluator_in_parallel
    if args.no_parallel:
        run_evaluator_in_parallel = False
    get_pool()
    fitness_methods = {
        "score": avg_of(training_seeds, "score"),
        "lines": avg_of(training_seeds, "lines"),
    }
    filename = f"save_{args.fitness_method}.pkl"
    ga = GA(100, 50, fitness_methods[args.fitness_method], filename)
    best = ga.run(resume=True)
    print("All Done")
    print(best)


def avg_of(seeds, result_key):
    def avg_of_inner(weights):
        results = []
        for seed in seeds:
            result = evaluate(seed, weights, run_evaluator_in_parallel)
            results.append(result[result_key])
        return statistics.mean(results)

    return avg_of_inner


def evaluate(seed: int, weights: Weights, parallel: bool = True):
    game = Game(seed, level=19)
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

    return {"lines": game.lines, "score": game.score}


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
