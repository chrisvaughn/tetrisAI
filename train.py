#!/usr/bin/env python
import argparse
from random import randint

from bot import GA, Evaluator, Weights, get_pool
from tetris import Game, Piece, Tetrominoes

run_evaluator_in_parallel = True


def nes_prng(value: int):
    bit1 = (value >> 1) & 1
    bit9 = (value >> 9) & 1
    lmb = bit1 ^ bit9
    return (lmb << 15) | (value >> 1)


def generate_piece_lists(num_pieces: int, seed: int = 0) -> list[Piece]:
    pieces = []
    for i in range(num_pieces):
        value = nes_prng(seed)
        seed = value
        p = Tetrominoes[value % len(Tetrominoes)].clone()
        p.set_position(6, 1)
        pieces.append(p)
    return pieces


def main(args):
    global run_evaluator_in_parallel
    if args.no_parallel:
        run_evaluator_in_parallel = False
    if run_evaluator_in_parallel:
        get_pool(args.num_of_parallel)
    piece_lists = []
    for i in range(args.num_iterations):
        piece_lists.append(generate_piece_lists(1000, randint(0, 10000000)))

    fitness_methods = {
        "score": top_3rd_avg_of(piece_lists, "score", args.scoring),
        "lines": top_3rd_avg_of(piece_lists, "lines", args.scoring),
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


def top_3rd_avg_of(piece_lists, result_key, scoring):
    def avg_of_inner(weights):
        results = []
        for piece_list in piece_lists:
            result = evaluate(weights, piece_list, run_evaluator_in_parallel, scoring)
            results.append(result[result_key])
        results = sorted(results)
        results = results[int(len(results) * 0.33) :]
        return sum(results) / len(results)

    return avg_of_inner


def evaluate(
    weights: Weights,
    piece_list: list[Piece],
    parallel: bool = True,
    scoring: str = "v2",
):
    game = Game(level=19, piece_list=piece_list)
    aie = Evaluator(game.state, weights, parallel, scoring)
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
        "--num-iterations",
        type=int,
        default=100,
        help="number of iterations to average top 3rd",
    )
    parser.add_argument(
        "--parallel-runners", dest="num_of_parallel", type=int, default=4
    )
    parser.add_argument("--scoring", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    main(args)
