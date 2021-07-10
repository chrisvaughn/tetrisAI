import time
from dataclasses import dataclass
from typing import List, Tuple, Union

from tetris import GameState

from .evaluation_pool import get_pool


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


class Evaluator:
    def __init__(self, state: GameState, weights: dict):
        self._initial_state = state
        self._weights = weights

    def update_state(self, state: GameState):
        self._initial_state = state

    def update_weights(self, weights: dict):
        self._weights = weights

    def execute_and_score(self, p: Tuple[int, int]) -> Move:
        state = self._initial_state.clone()
        execute_move(state, p[0], p[1])
        score, parameters = self.scoring_v1(state)
        move = Move(
            p[0],
            p[1],
            score,
            parameters,
            None,
            parameters["values"]["lines"],
        )
        return move

    def scoring_v1(self, state: GameState) -> Tuple[float, dict]:
        values = {
            "holes": state.count_holes(),
            "roughness": state.roughness(),
            "lines": state.check_full_lines(),
            "relative_height": state.relative_height(),
            "absolute_height": state.absolute_height(),
            "cumulative_height": state.cumulative_height(),
        }
        score = 0
        for k in self._weights.keys():
            score += self._weights[k] * values[k]
        return score, {"values": values, "weights": self._weights}

    def evaluate_all_moves(self) -> List[Move]:
        possible_moves: List[Move] = []
        meaningful_rotations = len(self._initial_state.current_piece.shapes)
        options = []
        for rot in range(meaningful_rotations):
            rotstate = self._initial_state.clone()
            for _ in range(rot):
                rotstate.rot_cw()
            l, r = rotstate.current_piece.possible_translations()
            for t in range(-l, r + 1):
                options.append((rot, t))

        imoves = get_pool().imap_unordered(self.execute_and_score, options)

        for m in imoves:
            possible_moves.append(m)

        return possible_moves

    def best_move(self, debug=False) -> Tuple[Move, float, int]:
        start = time.time()
        all_moves = sorted(
            self.evaluate_all_moves(),
            key=lambda x: (x.score, x.lines_completed),
            reverse=True,
        )
        if debug:
            print(all_moves)
        best_move = all_moves[0]
        end = time.time()
        return best_move, end - start, len(all_moves)
