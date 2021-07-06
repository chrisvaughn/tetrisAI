import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Union

from tetris import Board, GameState


def scoring_v1(state: GameState, weights) -> Tuple[float, dict]:
    values = {
        "holes": state.count_holes(),
        "roughness": state.roughness(),
        "lines": state.check_full_lines(),
        "relative_height": state.relative_height(),
        "absolute_height": state.absolute_height(),
        "cumulative_height": state.cumulative_height(),
    }
    score = 0
    for k in weights.keys():
        score += weights[k] * values[k]
    return score, {"values": values, "weights": weights}


def execute_move(state: GameState, rot: int, trans: int):
    for _ in range(rot):
        state.rot_cw()
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
    final_state: Union[GameState, None]
    lines_completed: int

    def to_sequence(self) -> List[str]:
        seq = []
        if self.rotations == 3:
            seq.append("rot_ccw")
        else:
            for i in range(self.rotations):
                seq.append("rot_cw")
        if self.translation < 0:
            for _ in range(abs(self.translation)):
                seq.append("move_left")
        if self.translation > 0:
            for _ in range(abs(self.translation)):
                seq.append("move_right")
        if not seq:
            seq.append("noop")
        return seq


class Evaluator:
    def __init__(self, state: GameState, weights: dict):
        self._initial_state = state
        self._weights = weights

    def evaluate_all_moves(
        self,
        initial_state: GameState,
        collect_final_state: bool = False,
    ) -> List[Move]:
        possible_moves: List[Move] = []
        meaningful_rotations = len(initial_state.current_piece.shapes)
        for rot in range(meaningful_rotations):
            rotstate = initial_state.clone()
            for _ in range(rot):
                rotstate.rot_cw()
            l, r = rotstate.current_piece.possible_translations()
            for t in range(-l, r + 1):
                state = initial_state.clone()
                execute_move(state, rot, t)
                score, parameters = scoring_v1(state, self._weights)
                move = Move(
                    rot,
                    t,
                    score,
                    parameters,
                    None,
                    parameters["values"]["lines"],
                )
                if collect_final_state:
                    move.final_state = state
                possible_moves.append(move)

        return possible_moves

    def best_move(
        self, collect_final_state=False, debug=False
    ) -> Tuple[Move, float, int]:
        start = time.time()
        all_moves = sorted(
            self.evaluate_all_moves(
                self._initial_state,
                collect_final_state=collect_final_state,
            ),
            key=lambda x: (x.score, x.lines_completed),
            reverse=True,
        )
        if debug:
            print(all_moves)
        best_move = all_moves[0]
        end = time.time()
        return best_move, end - start, len(all_moves)

    def random_valid_move_sequence(self) -> List[str]:
        sequence: List[str] = []
        for i in range(random.randint(0, 3)):
            if self._initial_state.rot_ccw_possible():
                self._initial_state.rot_ccw()
                sequence.append("rot_cw")

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

    def compare_initial_to_expected(self, expect_state: GameState) -> bool:
        if expect_state is None:
            print("Expected State is None, skipping diff")
            return True

        return self._initial_state.diff_state(expect_state)
