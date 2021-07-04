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
        state.rotate_cw()
    actual_t = trans
    if trans < 0:
        for i in range(1, abs(trans) + 1):
            if state.move_left():
                actual_t = -i
    if trans > 0:
        for i in range(1, trans + 1):
            if state.move_right():
                actual_t = i
    while state.move_down_possible():
        state.move_down()
    state.move_down()

    return actual_t


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
        lookahead: bool = True,
        collect_final_state: bool = False,
    ) -> List[Move]:
        possible_moves: List[Move] = []
        meaningful_rotations = len(initial_state.current_piece.shapes)
        for rot in range(meaningful_rotations + 1):
            for t in range(-5, 5):
                state = initial_state.clone()
                actual_t = execute_move(state, rot, t)

                if lookahead:
                    lookahead_state = state.clone()
                    # move next_piece to current_piece and do it again
                    next_piece_scores = []
                    lookahead_state.current_piece = lookahead_state.next_piece
                    x = math.floor(Board.columns / 2) - math.ceil(
                        len(lookahead_state.current_piece.shape[0]) / 2
                    )
                    lookahead_state.current_piece.set_position(x, 0)
                    meaningful_rotations = len(lookahead_state.current_piece.shapes)
                    for rot2 in range(meaningful_rotations + 1):
                        for t2 in range(-5, 5):
                            execute_move(lookahead_state, rot2, t2)
                            score, parameters = scoring_v1(
                                lookahead_state, self._weights
                            )
                            move = Move(
                                rot,
                                t,
                                score,
                                parameters,
                                None,
                                lookahead_state.check_full_lines(),
                            )
                            if collect_final_state:
                                move.final_state = state
                            next_piece_scores.append(move)
                    best_move_with_look_ahead = sorted(
                        next_piece_scores,
                        key=lambda x: x.score,
                        reverse=True,
                    )[0]
                    possible_moves.append(best_move_with_look_ahead)
                else:
                    score, parameters = scoring_v1(state, self._weights)
                    move = Move(
                        rot,
                        actual_t,
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
        self, lookahead=True, collect_final_state=False, debug=False
    ) -> Tuple[Move, float]:
        start = time.time()
        all_moves = sorted(
            self.evaluate_all_moves(
                self._initial_state,
                lookahead=lookahead,
                collect_final_state=collect_final_state,
            ),
            key=lambda x: (x.score, x.lines_completed),
            reverse=True,
        )
        if debug:
            print(all_moves)
        best_move = all_moves[0]
        end = time.time()
        return best_move, end - start

    def random_valid_move_sequence(self) -> List[str]:
        sequence: List[str] = []
        for i in range(random.randint(0, 3)):
            if self._initial_state.rot_ccw_possible():
                self._initial_state.rotate_ccw()
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
