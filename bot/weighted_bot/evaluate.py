import time
from dataclasses import dataclass
from typing import List, Tuple, Union

from tetris import MS_PER_FRAME, GameState, InvalidMove

from .evaluation_pool import get_pool


@dataclass
class Weights:
    holes: float = 0
    depth_weighted_holes: float = 0
    roughness: float = 0
    lines: float = 0
    relative_height: float = 0
    absolute_height: float = 0
    cumulative_height: float = 0
    well_count: float = 0
    well_cells: float = 0
    deep_well_count: float = 0
    total_cells: float = 0
    total_weighted_cells: float = 0
    row_transitions: float = 0
    move_cost: float = 0


@dataclass
class Move:
    rotations: int
    translation: int
    score: float
    score_parameters: dict
    final_state: Union[GameState, None]
    lines_completed: int
    end_state: GameState
    soft_drop_rows: int = 0

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
        seq = [(r,) for r in rotations] + [(t,) for t in translations]
        if not seq:
            seq.append(("noop",))
        return seq


def execute_move(state: GameState, rot: int, trans: int) -> int:
    """Apply rotation/translation then drop the piece straight down to lock.

    Returns the number of rows the piece fell during that final straight drop —
    the row count that would be awarded as soft-drop score in NES Tetris if the
    player held down continuously into the lock (the rotation/translation above
    represents positioning, and this drop is the final, unshifted "move down").
    """
    # Simulate piece falling while the player executes the move sequence.
    # rot=3 maps to 1 CCW keypress in practice (see BotMove.to_sequence).
    if state.frames_per_cell > 0:
        effective_rotations = 1 if rot == 3 else rot
        keypresses = effective_rotations + abs(trans)
        ms_per_row = state.frames_per_cell * MS_PER_FRAME
        rows_to_drop = int(keypresses * state.ms_per_keypress / ms_per_row)
        for _ in range(rows_to_drop):
            if not state.move_down_possible():
                break
            state.current_piece.move_down()

    if rot > 0 and not state.rot_cw(rot):
        raise InvalidMove(state.current_piece)
    if trans < 0 and not state.move_left(abs(trans)):
        raise InvalidMove(state.current_piece)
    if trans > 0 and not state.move_right(trans):
        raise InvalidMove(state.current_piece)

    # Fast drop: compute exact landing row from column peaks. Cells still
    # above the visible board (negative row) must be included too, since
    # they can be the ones that limit how far the piece can fall.
    px, py = state.current_piece.zero_based_corner_xy
    peaks = state.board._get_peaks()
    drop = min(int(peaks[px + cx]) - (py + cy) - 1 for cy, cx in state.current_piece.cell_tuples)
    drop = max(0, drop)
    if drop > 0:
        state.current_piece.move_down(drop)
    while state.move_down_possible():
        state.move_down()
    state.move_down()
    return drop


class Evaluator:
    def __init__(
        self,
        state: GameState,
        weights: Weights,
        parallel: bool = True,
        scoring: str = "v1",
        lookahead: bool = False,
        beam_width: int = None,
    ):
        self._initial_state = state
        self._weights = weights
        self.parallel = parallel
        self._lookahead = lookahead
        self._beam_width = beam_width
        if scoring == "v1":
            self.scoring_func = self.scoring_v1
        elif scoring == "v2":
            self.scoring_func = self.scoring_v2
        else:
            raise ValueError(f"Invalid scoring function {scoring}")

    def update_state(self, state: GameState):
        self._initial_state = state

    def update_weights(self, weights: Weights):
        self._weights = weights

    def execute_and_score(self, p: Tuple[int, int, int]) -> List[Move]:
        rot, left_trans, right_trans = p
        moves = []
        for trans in range(-left_trans, right_trans + 1):
            state = self._initial_state.clone()
            try:
                soft_drop_rows = execute_move(state, rot, trans)
            except InvalidMove:
                continue

            score, parameters = self.scoring_func(state)
            keypresses = (1 if rot == 3 else rot) + abs(trans)
            score += self._weights.move_cost * keypresses
            parameters["values"]["move_cost"] = keypresses
            move = Move(
                rot,
                trans,
                score,
                parameters,
                None,
                parameters["values"]["lines"],
                state,
                soft_drop_rows,
            )
            moves.append(move)
        return moves

    def scoring_v1(self, state: GameState) -> Tuple[float, dict]:
        holes, depth_weighted = state.count_holes()
        values = {
            "holes": holes,
            "depth_weighted_holes": depth_weighted,
            "roughness": state.roughness(),
            "lines": state.check_full_lines(),
            "relative_height": state.relative_height(),
            "absolute_height": state.absolute_height(),
            "cumulative_height": state.cumulative_height(),
            "well_count": state.deep_well_count(),
        }
        score = 0
        for k in values.keys():
            score += getattr(self._weights, k) * values[k]
        unreachable = state.unreachable_cells()
        if unreachable > 0:
            score -= 1000 * unreachable
        spawn_blocked = state.spawn_zone_filled()
        if spawn_blocked > 0:
            score -= 1000 * spawn_blocked
        return score, {"values": values, "weights": self._weights}

    def scoring_v2(self, state: GameState) -> Tuple[float, dict]:
        holes, depth_weighted = state.count_holes()
        cells, weighted_cells = state.count_cells()
        values = {
            "lines": state.check_full_lines(),
            "well_cells": state.well_cells(),
            "deep_well_count": state.deep_well_count(),
            "holes": holes,
            "depth_weighted_holes": depth_weighted,
            "cumulative_height": state.cumulative_height(),
            "roughness": state.roughness(),
            "row_transitions": state.row_transitions(),
            "relative_height": state.relative_height(),
            "absolute_height": state.absolute_height(),
            "total_cells": cells,
            "total_weighted_cells": weighted_cells,
        }
        score = 0
        for k in values.keys():
            score += getattr(self._weights, k) * values[k]
        unreachable = state.unreachable_cells()
        if unreachable > 0:
            score -= 1000 * unreachable
        spawn_blocked = state.spawn_zone_filled()
        if spawn_blocked > 0:
            score -= 1000 * spawn_blocked
        return score, {"values": values, "weights": self._weights}

    def evaluate_all_moves(self) -> List[Move]:
        possible_moves: List[Move] = []
        meaningful_rotations = len(self._initial_state.current_piece.shapes)
        options = []
        piece = self._initial_state.current_piece.clone()
        for rot in range(meaningful_rotations):
            if rot > 0:
                piece.rot_cw()
            left, r = piece.possible_translations()
            options.append((rot, left, r))

        if self.parallel:
            imoves = get_pool().imap_unordered(self.execute_and_score, options)
            for m in imoves:
                possible_moves.extend(m)
        else:
            for option in options:
                moves = self.execute_and_score(option)
                possible_moves.extend(moves)

        return possible_moves

    def _best_lookahead_score(self, move: Move) -> float:
        """Evaluate all placements of the next piece from move.end_state and return the best score."""
        state = move.end_state.clone()
        # end_state already has lines cleared (scoring_func calls check_full_lines).
        # If game is over or next piece unknown, fall back to the level-1 score.
        if state.next_piece is None or state.check_game_over():
            return move.score
        # Normalize next piece to spawn position/rotation regardless of detection source.
        next_p = state.next_piece.clone()
        next_p.set_position(6, 1)
        next_p.current_shape_idx = next_p.default_shape_idx
        state.current_piece = next_p
        state.next_piece = None
        best = float("-inf")
        piece = state.current_piece.clone()
        for rot in range(len(piece.shapes)):
            p = piece.clone()
            if rot > 0:
                p.rot_cw(rot)
            left, right = p.possible_translations()
            for trans in range(-left, right + 1):
                s = state.clone()
                try:
                    execute_move(s, rot, trans)
                except InvalidMove:
                    continue
                score2, _ = self.scoring_func(s)
                score2 += self._weights.move_cost * ((1 if rot == 3 else rot) + abs(trans))
                if score2 > best:
                    best = score2
        return best if best > float("-inf") else move.score

    def best_move(self, debug=False) -> Tuple[Move, float, int]:
        start = time.time()
        candidates = self.evaluate_all_moves()

        if self._lookahead and candidates:
            candidates.sort(key=lambda m: m.score, reverse=True)
            beam = candidates[: self._beam_width] if self._beam_width else candidates
            for move in beam:
                move.score = self._best_lookahead_score(move)
            candidates = beam

        all_moves = sorted(
            candidates,
            key=lambda x: (
                1 if x.lines_completed == 4 else 0,
                x.score,
                x.lines_completed,
                -abs(x.translation),
                -x.rotations,
            ),
            reverse=True,
        )
        if debug:
            print(all_moves)
        best_move = all_moves[0]
        end = time.time()
        return best_move, end - start, len(all_moves)
