#!/usr/bin/env python

import argparse
import os
import pickle
import time
from collections import Counter
from typing import Union

import cv2

from bot import WeightedBot, RandomBot, by_mode, get_pool
from vision import Detectorist
from tetris import Game, GameState, Tetrominoes


def get_bot(bot_model, save_file=None, save_gen=None):
    """Create a bot instance based on the bot-model argument"""
    if bot_model == "Random":
        return RandomBot("RandomBot")
    elif bot_model == "WeightedBotLines":
        weights = get_weights_from_save("lines", save_file, save_gen)
        return WeightedBot(weights, name="WeightedBotLines")
    elif bot_model == "WeightedBotScore":
        weights = get_weights_from_save("score", save_file, save_gen)
        return WeightedBot(weights, name="WeightedBotScore")
    else:
        raise ValueError(f"Unknown bot model: {bot_model}")


def get_weights_from_save(mode, save_file=None, save_gen=None):
    """Get weights from save file or use default weights"""
    if save_file and os.path.isfile(save_file):
        with open(save_file, "rb") as f:
            saved = pickle.load(f)
            if save_gen is not None:
                g = saved.best_for_each_generation[save_gen]
                print(f"Using Genome {g.id} from generation {save_gen}")
                return g.weights
            else:
                return saved.genomes[0].weights
    print(f"Getting weights for mode: {mode}")
    return by_mode[mode]


def print_final_stats(lines: int, piece_stats: Counter, combos: Counter):
    print(f"Lines Completed: {lines}")
    print(f"Piece Stats: Total {sum(piece_stats.values())}")
    for piece in Tetrominoes:
        print(f"\t{piece.name.upper()}: {piece_stats[piece.name]}")
    print("Line Combos:")
    for i in range(1, 5):
        print(f"\t{i}: {combos[i]}")


def main(args):
    # init evaluation pool
    get_pool()
    bot = get_bot(args.bot_model, args.save_file, args.save_gen)
    print(f"Using bot: {bot.name}")
    print(f"Bot stats: {bot.get_stats()}")
    if args.emulator:
        run_with_emulator(args, bot)
    else:
        run_in_memory(args, bot)


def run_in_memory(args, bot):
    seed = args.seed
    soft_drop = args.drop
    move_count = 0
    drop_enabled = False
    move_sequence = []
    game = Game(seed, args.level)
    game.display()
    game.start()
    while not game.game_over:
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        game.display()
        if game.state.new_piece() and not move_sequence:
            drop_enabled = False
            move_count += 1
            bot.update_state(game.state)
            best_move, time_taken, moves_considered = bot.get_best_move(debug=False)
            move_sequence = best_move.to_sequence()
            if args.stats:
                print(
                    f"Move {move_count}: Piece: {game.state.current_piece.name}, Considered {moves_considered} moves in {int(time_taken * 1000)} ms."
                )
                print(f"\tSequence: {move_sequence}")
        if move_sequence:
            moves = move_sequence.pop(0)
            for move in moves:
                if move != "noop":
                    getattr(game, move)()
            game.move_seq_complete()
            drop_enabled = True
        elif drop_enabled and soft_drop:
            game.move_down()

    print("Game Over")
    print_final_stats(game.lines, game.piece_stats, game.line_combos)


def run_with_emulator(args, bot):
    from emulator import Emulator

    emulator = Emulator(args.limit_speed, args.music, args.level, args.sound)
    soft_drop = args.drop
    move_sequence = []
    move_count = 0
    lines_completed = 0
    line_combos = Counter()
    piece_stats = Counter()

    gs = GameState(args.seed)
    detector = Detectorist()

    expected_state: Union[GameState, None] = None
    move_sequence_executed = False
    next_best_move, next_time_taken, next_moves_considered = None, None, None
    while True:
        screen = emulator.get_latest_image()
        if screen is None:
            time.sleep(0.01)
            continue
        detector.update(screen)
        if detector.board.game_over():
            break

        gs.update(detector.board, detector.current_piece, detector.next_piece)
        if gs.new_piece():
            if (
                expected_state is not None
                and not gs.board.compare(expected_state.board)
                or next_best_move is None
            ):
                if expected_state is not None and args.debug:
                    print("Board state not the same")
                    print("Expected:")
                    expected_state.board.print()
                    print("Actual:")
                    gs.board.print()
                    print("\n")
                # re-plan best move with current board state
                print("Re-planning best move")
                bot.update_state(gs)
                best_move, time_taken, moves_considered = bot.get_best_move(args.debug)
            else:
                best_move, time_taken, moves_considered = (
                    next_best_move,
                    next_time_taken,
                    next_moves_considered,
                )

            move_sequence_executed = False
            emulator.drop_off()
            move_count += 1
            piece_stats[gs.current_piece.name] += 1
            move_sequence = best_move.to_sequence()
            # Use the end state from the bot move
            expected_state = best_move.end_state.clone() if best_move.end_state else gs.clone()
            if args.stats:
                print(
                    f"Move {move_count}: Piece: {gs.current_piece.name}, Considered {moves_considered} moves in {int(time_taken * 1000)} ms."
                )
                print(f"\tSequence: {move_sequence}")
            # Track lines completed if available
            if hasattr(best_move, 'lines_completed') and best_move.lines_completed:
                lines = best_move.lines_completed
                line_combos[lines] += 1
                lines_completed += lines
        elif move_sequence_executed and next_best_move is None:
            # we can plan the next move based on the expected board state and the next piece
            next_piece = detector.detect_next_piece()
            if next_piece:
                # use best move computed with "next piece" and expected board state
                next_piece_gs = GameState()
                next_piece.set_position(6, 1)
                next_piece_gs.update(expected_state.board, next_piece, None)
                bot.update_state(next_piece_gs)
                next_best_move, next_time_taken, next_moves_considered = bot.get_best_move(
                    args.debug
                )

        if move_sequence:
            move_sequence_executed = True
            next_best_move = None
        while move_sequence:
            moves = move_sequence.pop(0)
            emulator.send_multiple_moves(moves)

        if soft_drop:
            emulator.drop_on()

    print("Game Over")
    print_final_stats(lines_completed, piece_stats, line_combos)
    emulator.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run tetris bot")
    parser.add_argument(
        "--emulator", action="store_true", default=False, help="run with emulator"
    )
    parser.add_argument(
        "--stats", action="store_true", default=False, help="print move stats display"
    )
    parser.add_argument(
        "--seed",
        default=int(time.time() * 100000),
        type=int,
        help="rng seed for non-emulator",
    )
    parser.add_argument(
        "--save-file",
        help="use weights from save file if present",
    )
    parser.add_argument(
        "--save-gen",
        type=int,
        help="use weights from specified generation in save file",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        default=False,
        help="soft drop pieces",
    )
    parser.add_argument(
        "--limit-speed",
        dest="limit_speed",
        action="store_true",
        default=False,
        help="applies cheat to emulator to limit speed to level 19",
    )
    parser.add_argument(
        "--sound", action="store_true", default=False, help="enable all sounds"
    )
    parser.add_argument(
        "--music", action="store_true", default=False, help="play music"
    )
    parser.add_argument("--level", default=19, type=int, help="level to start at")
    parser.add_argument(
        "--bot-model",
        default="WeightedBotLines",
        choices=["Random", "WeightedBotLines", "WeightedBotScore"],
        help="bot model to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="enable debug info to be logged",
    )
    parser.add_argument("--scoring", choices=["v1", "v2"], default="v2")

    args = parser.parse_args()
    main(args)
