#!/usr/bin/env python
import argparse

from bot import GA, Evaluator, Weights, get_pool
from tetris import Game

run_evaluator_in_parallel = True


def main(args):
    global run_evaluator_in_parallel
    if args.no_parallel:
        run_evaluator_in_parallel = False
    if run_evaluator_in_parallel:
        get_pool(4)
    iterations = args.num_iterations
    fitness_methods = {
        "score": avg_of(iterations, "score"),
        "lines": avg_of(iterations, "lines"),
    }
    if args.save_file:
        filename = args.save_file
    else:
        filename = f"save_{args.fitness_method}.pkl"
    ga = GA(
        args.population,
        args.generations,
        fitness_methods[args.fitness_method],
        filename,
    )
    best = ga.run(resume=True)
    print("All Done")
    print(best)


def avg_of(iterations, result_key):
    def avg_of_inner(weights):
        results = []
        for i in range(iterations):
            result = evaluate(weights, run_evaluator_in_parallel)
            results.append(result[result_key])
        return sum(results) / len(results)

    return avg_of_inner


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
    parser.add_argument(
        "--save-file",
        help="override save file name",
    )
    parser.add_argument("--population", "-p", type=int, default=100)
    parser.add_argument("--generations", "-g", type=int, default=100)
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to average"
    )
    args = parser.parse_args()
    main(args)
