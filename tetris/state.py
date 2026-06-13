from collections import deque
from typing import Union

import cv2
import numpy as np

from .board import Board
from .constants import MS_PER_KEYPRESS
from .pieces import Piece, Tetrominoes


class InvalidMove(Exception):
    def __init__(self, piece: Piece):
        self.piece = piece

    def __str__(self):
        return f"Invalid move: {self.piece}"


def nes_prng(value: int):
    # 16-bit right-shift LFSR, feedback taps at bits 1 and 9.
    # To seed from NES RAM: value = ($0017 << 8) | $0018  (NOT $0018 << 8 | $0017).
    bit1 = (value >> 1) & 1
    bit9 = (value >> 9) & 1
    lmb = bit1 ^ bit9
    return (lmb << 15) | (value >> 1)


class GameState:
    def __init__(self, seed: int = 0, piece_list=None):
        self.board: Union[Board, None] = None
        self.current_piece: Union[Piece, None] = None
        self.next_piece: Union[Piece, None] = None
        self._last_piece_y: int = 0
        self._completed_lines: int = 0
        self._last_rn: int = seed
        self._first_piece: bool = True
        self.piece_list = deque(piece_list) if piece_list is not None else None
        self.frames_per_cell: int = 0  # 0 = no fall simulation; set by Game or caller
        self.ms_per_keypress: float = MS_PER_KEYPRESS

    def select_next_piece(self) -> Piece:
        if self.piece_list:
            return self.piece_list.popleft()

        value = nes_prng(self._last_rn)
        self._last_rn = value
        p = Tetrominoes[value % len(Tetrominoes)].clone()
        p.set_position(6, 1)
        return p

    def display(self):
        block_size = 28
        virtual_board = np.zeros((Board.rows * block_size, Board.columns * block_size, 3), dtype=np.uint8)
        for y, cols in enumerate(self.board.board):
            for x, cell in enumerate(cols):
                cv2.rectangle(
                    virtual_board,
                    (x * block_size, y * block_size),
                    ((x + 1) * block_size, (y + 1) * block_size),
                    (255, 255, 255),
                    cv2.FILLED if cell != 0 else None,
                )
        if self.current_piece:
            px, py = self.current_piece.zero_based_corner_xy
            for (y, x), value in np.ndenumerate(self.current_piece.shape):
                if value != 0:
                    cv2.rectangle(
                        virtual_board,
                        (
                            (px + x) * block_size,
                            (py + y) * block_size,
                        ),
                        (
                            (px + x + 1) * block_size,
                            (py + y + 1) * block_size,
                        ),
                        (255, 0, 0),
                        cv2.FILLED,
                    )

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (block_size * 2, block_size * 2)
        font_scale = 1
        color = (255, 255, 0)
        thickness = 2

        cv2.putText(
            virtual_board,
            str(self._completed_lines),
            org,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.imshow("Virtual Board", virtual_board)

    def update(self, board: Board, current_piece: Piece, next_piece: Piece = None):
        self.board = board
        if self.current_piece:
            self._last_piece_y = self.current_piece.zero_based_corner_xy[1]
        if current_piece:
            self.current_piece = current_piece
        self.next_piece = next_piece

    def clone(self):
        new_state = GameState.__new__(GameState)
        new_state.board = self.board.clone()
        new_state.current_piece = self.current_piece.clone()
        new_state.next_piece = self.next_piece  # not mutated during move evaluation
        new_state._last_piece_y = self._last_piece_y
        new_state._completed_lines = self._completed_lines
        new_state._last_rn = self._last_rn
        new_state._first_piece = self._first_piece
        new_state.piece_list = self.piece_list  # not mutated during move evaluation
        new_state.frames_per_cell = self.frames_per_cell
        new_state.ms_per_keypress = self.ms_per_keypress
        return new_state

    def new_piece(self) -> bool:
        if self.current_piece and self._first_piece:
            self._first_piece = False
            return True
        if self.current_piece:
            return self.current_piece.zero_based_corner_xy[1] < self._last_piece_y
        return False

    def move_down(self, moves: int = 1):
        if self.move_down_possible(moves):
            self.current_piece.move_down(moves)
            return True
        else:
            if moves == 1:
                self.place_piece_on_board()
            return False

    def move_left(self, moves: int = 1):
        if self.move_left_possible(moves):
            self.current_piece.move_left(moves)
            return True
        return False

    def move_right(self, moves: int = 1) -> bool:
        if self.move_right_possible(moves):
            self.current_piece.move_right(moves)
            return True
        return False

    def move_down_possible(self, moves: int = 1) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        py_new = py + moves
        board = self.board.board
        for cy, cx in self.current_piece.cell_tuples:
            y = py_new + cy
            if y < 0:
                continue
            x = px + cx
            if y >= Board.rows or x < 0 or x >= Board.columns or board[y, x]:
                return False
        return True

    def move_left_possible(self, moves: int = 1) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        px_new = px - moves
        board = self.board.board
        for cy, cx in self.current_piece.cell_tuples:
            y = py + cy
            if y < 0:
                continue
            x = px_new + cx
            if x < 0 or board[y, x]:
                return False
        return True

    def move_right_possible(self, moves: int = 1) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        px_new = px + moves
        board = self.board.board
        for cy, cx in self.current_piece.cell_tuples:
            y = py + cy
            if y < 0:
                continue
            x = px_new + cx
            if x >= Board.columns or board[y, x]:
                return False
        return True

    def rot_ccw_possible(self, rot: int = 1):
        new_idx = (self.current_piece.current_shape_idx - rot) % len(self.current_piece.shapes)
        return self._rot_idx_possible(new_idx)

    def rot_cw_possible(self, rot: int = 1):
        new_idx = (self.current_piece.current_shape_idx + rot) % len(self.current_piece.shapes)
        return self._rot_idx_possible(new_idx)

    def _rot_idx_possible(self, shape_idx: int) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        board = self.board.board
        for cy, cx in self.current_piece.cell_tuples_for_rotation(shape_idx):
            y = py + cy
            if y < 0:
                continue
            x = px + cx
            if y >= Board.rows or x < 0 or x >= Board.columns or board[y, x]:
                return False
        return True

    def rot_possible(self, shape) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        board = self.board.board
        for cy, cx in zip(*np.nonzero(shape)):
            y = int(py + cy)
            if y < 0:
                continue
            x = int(px + cx)
            if y >= Board.rows or x < 0 or x >= Board.columns or board[y, x]:
                return False
        return True

    def rot_ccw(self, rot: int = 1) -> bool:
        if self.rot_ccw_possible(rot):
            self.current_piece.rot_ccw(rot)
            return True
        return False

    def rot_cw(self, rot: int = 1):
        if self.rot_cw_possible(rot):
            self.current_piece.rot_cw(rot)
            return True
        return False

    def place_piece_on_board(self):
        px, py = self.current_piece.zero_based_corner_xy
        board = self.board.board
        for cy, cx in self.current_piece.cell_tuples:
            y = py + cy
            if y >= 0:
                board[y, px + cx] = 1
        self.board.updated()

    def check_game_over(self, piece: Piece = None) -> bool:
        """Check whether `piece` (default: next_piece) would collide with the board
        at its spawn position, matching NES Tetris's spawn-time game-over check.

        Falls back to a row-0 occupancy check if no piece is available.
        """
        if piece is None:
            piece = self.next_piece
        if piece is None:
            return bool(np.any(self.board.board[0] != 0))

        board = self.board.board
        shape = piece.shapes[piece.default_shape_idx]
        for cy, cx in zip(*np.nonzero(shape)):
            y = int(cy) - 2
            x = int(cx) + 3
            if y < 0:
                continue
            if board[y, x]:
                return True
        return False

    def check_full_lines(self) -> int:
        full_row_mask = np.all(self.board.board != 0, axis=1)
        full_lines = int(full_row_mask.sum())
        if full_lines:
            removed_lines = self.board.board[full_row_mask]
            removed_lines.fill(0)
            clean_board = self.board.board[~full_row_mask]
            self.board.board = np.vstack((removed_lines, clean_board))
            self.board.updated()
        self._completed_lines += full_lines
        return full_lines

    def count_holes(self) -> (int, int):
        return self.board.count_holes()

    def cumulative_height(self) -> int:
        return self.board.cumulative_height()

    def roughness(self) -> int:
        return self.board.roughness()

    def relative_height(self) -> int:
        return self.board.relative_height()

    def absolute_height(self) -> int:
        return self.board.height()

    def deep_well_count(self) -> int:
        return self.board.deep_well_count()

    def well_cells(self) -> int:
        return self.board.count_well_cells()

    def count_cells(self) -> (int, int):
        return self.board.count_cells()

    def row_transitions(self) -> int:
        return self.board.count_row_transitions()

    def spawn_zone_filled(self) -> int:
        return self.board.spawn_zone_filled()

    def unreachable_cells(self) -> int:
        return self.board.count_unreachable_cells()
