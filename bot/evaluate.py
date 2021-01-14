import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

from tetris import Board, GameState


def scoring_v1(state: GameState, weights) -> Tuple[float, dict]:
    values = {
        "holes": state.count_holes(),
        "roughness": state.roughness(),
        "lines": state.check_full_lines(),
        "height": state.cumulative_height(),
    }
    score = 0
    for k in weights.keys():
        score += weights[k] * values[k]
    return score, {"values": values, "weights": weights}


def execute_move(state: GameState, rot: int, trans: int):
    for _ in range(rot):
        state.rotate_ccw()
    if trans < 0:
        for _ in range(abs(trans)):
            state.move_left()
    if trans > 0:
        for _ in range(trans):
            state.move_right()
    while state.move_down_possible():
        state.move_down()
    state.move_down()


@dataclass
class Move:
    rotations: int
    translation: int
    score: float
    score_parameters: dict

    def to_sequence(self) -> List[str]:
        seq = []
        for i in range(self.rotations):
            seq.append("rot_ccw")
        if self.translation < 0:
            for _ in range(abs(self.translation)):
                seq.append("move_left")
        if self.translation > 0:
            for _ in range(abs(self.translation)):
                seq.append("move_right")
        return seq


class Evaluator:
    def __init__(self, state: GameState, weights: dict):
        self._initial_state = state
        self._weights = weights

    def evaluate_all_moves(self, initial_state: GameState) -> List[Move]:
        possible_moves: List[Move] = []
        meaningful_rotations = initial_state.current_piece.valid_rotations + 1
        for rot in range(meaningful_rotations):
            for t in range(-5, 5):
                state = initial_state.clone()
                execute_move(state, rot, t)

                # move next_piece to current_piece and do it again
                next_piece_scores = []
                state.current_piece = state.next_piece
                x = math.floor(Board.columns / 2) - math.ceil(
                    len(state.current_piece.shape[0]) / 2
                )
                state.current_piece.set_position(x, 0)
                meaningful_rotations = state.current_piece.valid_rotations + 1
                for rot2 in range(meaningful_rotations):
                    for t2 in range(-5, 5):
                        execute_move(state, rot, t)

                        score, parameters = scoring_v1(state, self._weights)
                        next_piece_scores.append(Move(rot, t, score, parameters))
                best_move_with_look_ahead = sorted(
                    next_piece_scores,
                    key=lambda x: x.score,
                    reverse=True,
                )[0]
                possible_moves.append(best_move_with_look_ahead)
        return possible_moves

    def best_move(self) -> Tuple[Move, float]:
        start = time.time()
        all_moves = sorted(
            self.evaluate_all_moves(self._initial_state),
            key=lambda x: x.score,
            reverse=True,
        )
        best_move = all_moves[0]
        end = time.time()
        return best_move, end - start

    def random_valid_move_sequence(self) -> List[str]:
        sequence: List[str] = []
        for i in range(random.randint(0, 3)):
            if self._initial_state.rot_ccw_possible():
                self._initial_state.rotate_ccw()
                sequence.append("rot_ccw")

        left = random.random() > 0.5
        for i in range(random.randint(0, 5)):
            if left:
                if self._initial_state.move_left_possible():
                    self._initial_state.move_left()
                    sequence.append("move_left")
            else:
                if self._initial_state.move_right_possible():
                    self._initial_state.move_right()
                    sequence.append("move_right")

        while self._initial_state.move_down_possible():
            self._initial_state.move_down()
            sequence.append("move_down")

        return sequence
