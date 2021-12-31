import time
from dataclasses import dataclass
from itertools import zip_longest
from typing import List, Tuple, Union

from tetris import GameState, InvalidMove

from .evaluation_pool import get_pool


@dataclass
class Weights:
    holes: float = 0
    roughness: float = 0
    lines: float = 0
    relative_height: float = 0
    absolute_height: float = 0
    cumulative_height: float = 0
    well_count: float = 0
    movements_required: float = 0


@dataclass
class Move:
    rotations: int
    translation: int
    score: float
    score_parameters: dict
    final_state: Union[GameState, None]
    lines_completed: int

    def to_sequence(self) -> List[Tuple[str]]:
        rotations = []
        translations = []

        if self.rotations == 3:
            rotations.append("rot_ccw")
        else:
            for i in range(self.rotations):
                rotations.append("rot_cw")
        if self.translation < 0:
            for _ in range(abs(self.translation)):
                translations.append("move_left")
        if self.translation > 0:
            for _ in range(abs(self.translation)):
                translations.append("move_right")
        seq = list(zip_longest(rotations, translations, fillvalue="noop"))
        if not seq:
            seq.append(("noop",))
        return seq


def execute_move(state: GameState, rot: int, trans: int):
    state.rot_cw(rot)
    if trans < 0:
        state.move_left(abs(trans))
    if trans > 0:
        state.move_right(trans)
    # move down as much as possible in one go
    _, y = state.current_piece.zero_based_corner_xy
    y = y + state.current_piece.shape.shape[0]
    moves = state.board.rows - y - state.absolute_height() - 1
    if moves > 0:
        state.move_down(moves)
    # move one space at a time to check for collisions
    while state.move_down_possible():
        state.move_down()
    state.move_down()


class Evaluator:
    def __init__(self, state: GameState, weights: Weights, parallel: bool = True):
        self._initial_state = state
        self._weights = weights
        self.parallel = parallel

    def update_state(self, state: GameState):
        self._initial_state = state

    def update_weights(self, weights: Weights):
        self._weights = weights

    def execute_and_score(self, p: Tuple[int, int]) -> Union[Move, None]:
        state = self._initial_state.clone()
        try:
            execute_move(state, p[0], p[1])
        except InvalidMove:
            return None

        score, parameters = self.scoring_v1(state, p)
        move = Move(
            p[0],
            p[1],
            score,
            parameters,
            None,
            parameters["values"]["lines"],
        )
        return move

    def scoring_v1(
        self, state: GameState, movements_required: Tuple[int, int]
    ) -> Tuple[float, dict]:
        values = {
            "holes": state.count_holes(),
            "roughness": state.roughness(),
            "lines": state.check_full_lines(),
            "relative_height": state.relative_height(),
            "absolute_height": state.absolute_height(),
            "cumulative_height": state.cumulative_height(),
            "well_count": state.well_count(),
            "movements_required": sum(movements_required),
        }
        score = 0
        for k in values.keys():
            score += getattr(self._weights, k) * values[k]
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

        if self.parallel:
            imoves = get_pool().imap_unordered(self.execute_and_score, options)
            for m in imoves:
                if m is not None:
                    possible_moves.append(m)
        else:
            for option in options:
                move = self.execute_and_score(option)
                if move is not None:
                    possible_moves.append(move)

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
