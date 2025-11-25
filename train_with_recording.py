#!/usr/bin/env python
"""
Enhanced training script with game state recording for visualization.

Usage:
    # Basic training with recording
    python train_with_recording.py --fitness lines --population 100 --generations 50

    # With custom recording directory
    python train_with_recording.py --recording-dir ./my_recordings --snapshot-interval 5

    # View recordings after training
    python visualization/replay_viewer.py ./recordings --evolution --max-gen 50
"""
import argparse
import time
from pathlib import Path
from random import randint

from bot import GA, RandomBot, WeightedBot, Weights, get_pool
from tetris import Game, GameRecorder, Piece, Tetrominoes, TrainingRecorder

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

    # Setup recording
    recording_dir = Path(args.recording_dir)
    training_recorder = TrainingRecorder(
        recording_dir, snapshot_interval=args.snapshot_interval
    )

    print(f"Training {args.bot_model} model with recording...")
    print(f"Fitness method: {args.fitness_method}")
    print(f"Scoring: {args.scoring}")
    print(f"Population: {args.population}, Generations: {args.generations}")
    print(f"Parallel: {run_evaluator_in_parallel}")
    print(f"Recording to: {recording_dir}")
    print(f"Snapshot interval: every {args.snapshot_interval} pieces")
    print("-" * 50)

    # Generate piece lists for evaluation
    piece_lists = []
    for i in range(args.num_iterations):
        piece_lists.append(generate_piece_lists(1000, randint(0, 10000000)))

    # Create fitness function with recording
    fitness_methods = {
        "score": create_fitness_with_recording(
            piece_lists, "score", args.scoring, args.bot_model, training_recorder
        ),
        "lines": create_fitness_with_recording(
            piece_lists, "lines", args.scoring, args.bot_model, training_recorder
        ),
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
    best = ga.run(resume=args.resume)
    print("All Done")
    print(best)
    print(f"\nRecordings saved to: {recording_dir}")
    print(f"\nTo view recordings, run:")
    print(f"  python visualization/replay_viewer.py {recording_dir} --evolution")


def create_bot(
    bot_model: str, weights: Weights = None, parallel: bool = True, scoring: str = "v2"
):
    """Create a bot instance based on the bot model"""
    if bot_model == "WeightedBot":
        if weights is None:
            weights = Weights()
        return WeightedBot(weights, parallel=parallel, scoring=scoring)
    elif bot_model == "RandomBot":
        return RandomBot("RandomBot")
    else:
        raise ValueError(f"Unknown bot model: {bot_model}")


def create_fitness_with_recording(
    piece_lists, result_key, scoring, bot_model, training_recorder
):
    """Create fitness function that also records game states."""
    # Create a single bot instance to reuse
    bot = create_bot(bot_model, parallel=run_evaluator_in_parallel, scoring=scoring)

    # Track current generation and genome
    current_generation = [0]
    current_genome_id = [0]
    evaluations_this_gen = [0]

    def fitness_with_recording(weights):
        # Update the bot's weights
        if hasattr(bot, "update_weights"):
            bot.update_weights(weights)

        results = []

        # Only record first game of each genome to save space
        # You can change this to record all games if needed
        record_this = evaluations_this_gen[0] == 0

        for i, piece_list in enumerate(piece_lists):
            if record_this and i == 0:  # Record only first piece list
                game_id = f"gen{current_generation[0]:04d}_genome{current_genome_id[0]:04d}"
                result, recorder = evaluate_with_recording(
                    bot,
                    piece_list,
                    game_id=game_id,
                    generation=current_generation[0],
                    genome_id=current_genome_id[0],
                    training_recorder=training_recorder,
                )
            else:
                result = evaluate_without_recording(bot, piece_list)

            results.append(result[result_key])

        # Sort and take top 33%
        results = sorted(results)
        results = results[int(len(results) * 0.33) :]
        fitness = sum(results) / len(results)

        # Update counters
        evaluations_this_gen[0] += 1
        if evaluations_this_gen[0] >= 100:  # Assuming population size ~100
            evaluations_this_gen[0] = 0
            current_generation[0] += 1
            current_genome_id[0] = 0
        else:
            current_genome_id[0] += 1

        return fitness

    return fitness_with_recording


def evaluate_with_recording(
    bot: WeightedBot,
    piece_list: list[Piece],
    game_id: str,
    generation: int,
    genome_id: int,
    training_recorder: TrainingRecorder,
):
    """Evaluate a piece list with full recording."""
    game = Game(level=19, piece_list=piece_list)

    # Create recorder
    recorder = training_recorder.create_recorder(
        game_id=game_id,
        seed=0,  # Piece list is predetermined
        level=19,
        bot_name=bot.name,
        generation=generation,
        genome_id=genome_id,
    )

    drop_enabled = False
    move_count = 0
    move_sequence = []
    game.start()

    while not game.game_over:
        recorder.increment_frame()

        if game.state.new_piece() and not move_sequence:
            drop_enabled = False
            move_count += 1
            recorder.increment_piece()

            bot.update_state(game.state)
            best_move, time_taken, moves_considered = bot.get_best_move(debug=False)
            move_sequence = best_move.to_sequence()

            # Record state with move info
            move_info = {
                "rotations": best_move.rotations,
                "translation": best_move.translation,
                "score": best_move.score,
                "evaluation_time_ms": time_taken * 1000,
                "moves_considered": moves_considered,
            }

            recorder.record_state(
                game.state, game.score, game.lines, game.level, move_info=move_info
            )

        if move_sequence:
            moves = move_sequence.pop(0)
            for move in moves:
                if move != "noop":
                    getattr(game, move)()
            game.move_seq_complete()
            drop_enabled = True
        elif drop_enabled:
            game.move_down()

    # Finalize recording
    recorder.finalize(game.lines, game.score, game.piece_stats, game.line_combos)

    # Save recording
    training_recorder.save_recorder(recorder, generation, genome_id)

    return {"lines": game.lines, "score": game.score}, recorder


def evaluate_without_recording(bot: WeightedBot, piece_list: list[Piece]):
    """Fast evaluation without recording (for non-best genomes)."""
    game = Game(level=19, piece_list=piece_list)
    drop_enabled = False
    move_sequence = []
    game.start()

    while not game.game_over:
        if game.state.new_piece() and not move_sequence:
            drop_enabled = False
            bot.update_state(game.state)
            best_move, _, _ = bot.get_best_move(debug=False)
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
    parser = argparse.ArgumentParser(
        description="train a tetris bot with game recording"
    )
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
    parser.add_argument(
        "--bot-model",
        choices=["WeightedBot", "RandomBot"],
        default="WeightedBot",
        help="bot model to train",
    )
    parser.add_argument(
        "--recording-dir",
        type=str,
        default="./recordings",
        help="directory to save game recordings",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=1,
        help="record every N pieces (1=every piece, 5=every 5th piece)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from save file if exists",
    )
    args = parser.parse_args()
    main(args)
