#!/usr/bin/env python
import argparse
import os
import pickle
from random import randint

from bot import GA, RandomBot, WeightedBot, Weights
from tetris import Game, Piece, Tetrominoes


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


class GenomeFitness:
    """Picklable fitness callable for genome-level parallel evaluation."""

    def __init__(self, piece_lists, result_key, scoring, bot_model, max_lines=None):
        self.piece_lists = piece_lists
        self.result_key = result_key
        self.scoring = scoring
        self.bot_model = bot_model
        self.max_lines = max_lines

    def __call__(self, weights):
        bot = create_bot(self.bot_model, weights=weights, parallel=False, scoring=self.scoring)
        results = []
        for piece_list in self.piece_lists:
            result = evaluate_with_bot(bot, piece_list, max_lines=self.max_lines)
            results.append(result[self.result_key])
        results = sorted(results)
        results = results[int(len(results) * 0.67) :]
        return sum(results) / len(results)


def main(args):
    genome_workers = 1 if args.no_parallel else args.num_of_parallel

    print(f"Training {args.bot_model} model...")
    print(f"Fitness method: {args.fitness_method}")
    print(f"Scoring: {args.scoring}")
    print(f"Population: {args.population}, Generations: {args.generations}")
    print(f"Genome workers: {genome_workers}")
    print("-" * 50)

    if args.save_file:
        filename = args.save_file
    else:
        filename = f"save_{args.fitness_method}.pkl"

    piece_lists = []
    if os.path.isfile(filename):
        # Save files are written by this same process — trusted local data only.
        with open(filename, "rb") as f:
            save = pickle.load(f)
        piece_lists = getattr(save, "piece_lists", None) or []
        if piece_lists:
            print(f"Using {len(piece_lists)} piece lists from save file")
    if not piece_lists:
        for i in range(args.num_iterations):
            piece_lists.append(generate_piece_lists(1000, randint(0, 10000000)))

    max_lines = args.max_lines or None
    fitness_methods = {
        "score": top_two_thirds_avg_of(piece_lists, "score", args.scoring, args.bot_model, max_lines),
        "lines": top_two_thirds_avg_of(piece_lists, "lines", args.scoring, args.bot_model, max_lines),
    }
    ga = GA(
        args.population,
        args.generations,
        fitness_methods[args.fitness_method],
        filename,
        command_args=vars(args),
        genome_workers=genome_workers,
        piece_lists=piece_lists,
    )
    best = ga.run(resume=True)
    print("All Done")
    print(best)


def create_bot(bot_model: str, weights: Weights = None, parallel: bool = True, scoring: str = "v2"):
    if bot_model == "WeightedBot":
        if weights is None:
            weights = Weights()
        return WeightedBot(weights, parallel=parallel, scoring=scoring)
    elif bot_model == "RandomBot":
        return RandomBot("RandomBot")
    else:
        raise ValueError(f"Unknown bot model: {bot_model}")


def top_two_thirds_avg_of(piece_lists, result_key, scoring, bot_model, max_lines=None):
    return GenomeFitness(piece_lists, result_key, scoring, bot_model, max_lines)


def evaluate_with_bot(bot: WeightedBot, piece_list: list[Piece], max_lines: int = None):
    game = Game(level=19, piece_list=piece_list)
    drop_enabled = False
    move_count = 0
    move_sequence = []
    game.start()
    while not game.game_over:
        if max_lines and game.lines >= max_lines:
            break
        if game.state.new_piece() and not move_sequence:
            drop_enabled = False
            move_count += 1
            bot.update_state(game.state)
            best_move, time_taken, moves_considered = bot.get_best_move(debug=False)
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


def evaluate(
    weights: Weights,
    piece_list: list[Piece],
    parallel: bool = True,
    scoring: str = "v2",
    bot_model: str = "WeightedBot",
):
    """Legacy evaluate function for backward compatibility"""
    bot = create_bot(bot_model, weights, parallel, scoring)
    return evaluate_with_bot(bot, piece_list)


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
    parser.add_argument("--parallel-runners", dest="num_of_parallel", type=int, default=8)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=230,
        help="stop each evaluation game after this many lines cleared (0 = no cap)",
    )
    parser.add_argument("--scoring", choices=["v1", "v2"], default="v2")
    parser.add_argument(
        "--bot-model", choices=["WeightedBot", "RandomBot"], default="WeightedBot", help="bot model to train"
    )
    args = parser.parse_args()
    main(args)
