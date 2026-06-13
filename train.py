#!/usr/bin/env python
import argparse
import os
import pickle
from random import randint

import numpy as np

from bot import GA, RandomBot, WeightedBot, Weights
from bot.weighted_bot.defined_weights import by_mode as defined_weights_by_mode
from tetris import Board, Game, GameState, Piece, Tetrominoes, frames_per_cell_by_level
from tetris.game import score_by_number_of_lines_cleared


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

    def __init__(
        self,
        piece_lists,
        result_key,
        scoring,
        bot_model,
        max_lines=None,
        lookahead=False,
        beam_width=None,
        fitness_fraction=0.33,
        soft_drop_score=False,
    ):
        self.piece_lists = piece_lists
        self.result_key = result_key
        self.scoring = scoring
        self.bot_model = bot_model
        self.max_lines = max_lines
        self.lookahead = lookahead
        self.beam_width = beam_width
        self.fitness_fraction = fitness_fraction
        self.soft_drop_score = soft_drop_score

    def __call__(self, weights):
        bot = create_bot(
            self.bot_model,
            weights=weights,
            parallel=False,
            scoring=self.scoring,
            lookahead=self.lookahead,
            beam_width=self.beam_width,
        )
        results = []
        for piece_list in self.piece_lists:
            result = simulate_game(bot, piece_list, max_lines=self.max_lines, soft_drop_score=self.soft_drop_score)
            results.append(result[self.result_key])
        results = sorted(results)
        results = results[int(len(results) * (1 - self.fitness_fraction)) :]
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
        filename = os.path.join("saves", f"save_{args.fitness_method}.pkl")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

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

    seed_weights = []
    for mode in args.seed_builtin or []:
        if mode in defined_weights_by_mode:
            seed_weights.append(defined_weights_by_mode[mode])
            print(f"Seeding from built-in weights: {mode}")
    for seed_file in args.seed_file or []:
        if os.path.isfile(seed_file):
            # Save files are written by this same process — trusted local data only.
            with open(seed_file, "rb") as f:
                seed_save = pickle.load(f)
            best = max(seed_save.best_for_each_generation, key=lambda g: g.fitness)
            seed_weights.append(best.weights)
            print(f"Seeding from {seed_file}: genome id={best.id} fitness={best.fitness:.4f}")
    if seed_weights:
        n_seeded = len(seed_weights) * args.seeds_per_genome
        print(f"{len(seed_weights)} seed(s) × {args.seeds_per_genome} variants = {n_seeded} seeded genomes")

    fitness_methods = {
        "score": top_two_thirds_avg_of(
            piece_lists,
            "score",
            args.scoring,
            args.bot_model,
            max_lines,
            args.lookahead,
            args.beam_width,
            args.fitness_fraction,
            args.soft_drop_score,
        ),
        "lines": top_two_thirds_avg_of(
            piece_lists,
            "lines",
            args.scoring,
            args.bot_model,
            max_lines,
            args.lookahead,
            args.beam_width,
            args.fitness_fraction,
        ),
    }
    ga = GA(
        args.population,
        args.generations,
        fitness_methods[args.fitness_method],
        filename,
        command_args=vars(args),
        genome_workers=genome_workers,
        piece_lists=piece_lists,
        seed_weights=seed_weights,
        seeds_per_genome=args.seeds_per_genome,
        seed_noise=args.seed_noise,
        restart_noise=args.restart_noise,
        restart_random_count=args.restart_random_count,
    )
    best = ga.run(resume=True)
    print("All Done")
    print(best)


def create_bot(
    bot_model: str,
    weights: Weights = None,
    parallel: bool = True,
    scoring: str = "v2",
    lookahead: bool = False,
    beam_width: int = None,
):
    if bot_model == "WeightedBot":
        if weights is None:
            weights = Weights()
        return WeightedBot(weights, parallel=parallel, scoring=scoring, lookahead=lookahead, beam_width=beam_width)
    elif bot_model == "RandomBot":
        return RandomBot("RandomBot")
    else:
        raise ValueError(f"Unknown bot model: {bot_model}")


def top_two_thirds_avg_of(
    piece_lists,
    result_key,
    scoring,
    bot_model,
    max_lines=None,
    lookahead=False,
    beam_width=None,
    fitness_fraction=0.33,
    soft_drop_score=False,
):
    return GenomeFitness(
        piece_lists,
        result_key,
        scoring,
        bot_model,
        max_lines,
        lookahead,
        beam_width,
        fitness_fraction,
        soft_drop_score,
    )


def simulate_game(bot, piece_list: list[Piece], max_lines: int = None, level: int = 19, soft_drop_score: bool = False):
    """Fast synchronous game simulation for training — no threading, no sleeping."""
    state = GameState(piece_list=list(piece_list))
    state.board = Board()
    state.frames_per_cell = frames_per_cell_by_level[level]

    lines = 0
    score = 0
    starting_level = level
    first_advance_lines = min(level * 10 + 10, max(100, level * 10 - 50))

    cp = state.select_next_piece()
    next_p = state.piece_list[0].clone() if state.piece_list else None
    if next_p:
        next_p.set_position(6, 1)
    state.update(state.board, cp, next_p)

    while True:
        if max_lines and lines >= max_lines:
            break
        if not state.piece_list:
            break

        bot.update_state(state)
        try:
            best_move, _, _ = bot.get_best_move(debug=False)
        except IndexError:
            break
        end = best_move.end_state

        if np.any(end.board.board[0] != 0):
            break

        cleared = best_move.lines_completed
        if cleared > 0:
            lines += cleared
            if lines >= first_advance_lines:
                new_level = starting_level + 1 + (lines - first_advance_lines) // 10
                if new_level != level:
                    level = new_level
                    state.frames_per_cell = frames_per_cell_by_level.get(level, 1)
            if cleared <= len(score_by_number_of_lines_cleared):
                score += score_by_number_of_lines_cleared[cleared - 1] * (level + 1)

        if soft_drop_score:
            score += best_move.soft_drop_rows

        state.board = end.board
        cp = state.select_next_piece()
        next_p = state.piece_list[0].clone() if state.piece_list else None
        if next_p:
            next_p.set_position(6, 1)
        state.update(state.board, cp, next_p)

    return {"lines": lines, "score": score}


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
        "--soft-drop-score",
        dest="soft_drop_score",
        action="store_true",
        default=False,
        help="award NES soft-drop points (1 per row of the final drop) for --fitness score",
    )
    parser.add_argument(
        "--bot-model", choices=["WeightedBot", "RandomBot"], default="WeightedBot", help="bot model to train"
    )
    parser.add_argument(
        "--lookahead", action="store_true", default=False, help="evaluate next piece for each candidate move"
    )
    parser.add_argument(
        "--beam-width",
        dest="beam_width",
        type=int,
        default=None,
        help="limit lookahead to top-N level-1 candidates (default: all)",
    )
    parser.add_argument(
        "--fitness-fraction",
        dest="fitness_fraction",
        type=float,
        default=0.33,
        help="top fraction of games used for fitness (default: 0.33)",
    )
    parser.add_argument(
        "--seed-file",
        dest="seed_file",
        action="append",
        metavar="PATH",
        help="seed population from best genome in save file (repeatable)",
    )
    parser.add_argument(
        "--seed-builtin",
        dest="seed_builtin",
        action="append",
        choices=["lines", "score"],
        metavar="{lines,score}",
        help="seed population from built-in defined weights (repeatable)",
    )
    parser.add_argument(
        "--seeds-per-genome",
        dest="seeds_per_genome",
        type=int,
        default=5,
        help="variants to generate per seed genome (default: 5)",
    )
    parser.add_argument(
        "--seed-noise",
        dest="seed_noise",
        type=float,
        default=0.3,
        help="gauss std for seed variation (default: 0.3)",
    )
    parser.add_argument(
        "--restart-noise",
        dest="restart_noise",
        type=float,
        default=2.0,
        help="gauss std for restart variants (default: 2.0)",
    )
    parser.add_argument(
        "--restart-random",
        dest="restart_random_count",
        type=int,
        default=20,
        help="fresh random genomes injected on each restart (default: 20)",
    )
    args = parser.parse_args()
    main(args)
