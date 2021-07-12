import copy
from typing import Union

import cv2
import numpy as np

from .board import Board
from .pieces import Piece, Tetrominoes


def nes_prng(value: int):
    bit1 = (value >> 1) & 1
    bit9 = (value >> 9) & 1
    lmb = bit1 ^ bit9
    return (lmb << 15) | (value >> 1)


class GameState:
    def __init__(
        self,
        board: Board,
        current_piece: Union[Piece, None],
        next_piece: Union[Piece, None] = None,
        seed: int = 0
    ):
        self.board = board
        self.current_piece = current_piece
        self.next_piece = next_piece
        self._last_piece: Union[Piece, None] = None
        self._completed_lines: int = 0
        self._last_rn: int = seed

    def select_next_piece(self) -> Piece:
        value = nes_prng(self._last_rn)
        self._last_rn = value
        p = Tetrominoes[value % len(Tetrominoes)]
        p.set_position(6, 1)
        return p

    def display(self):
        block_size = 28
        virtual_board = np.zeros(
            (Board.rows * block_size, Board.columns * block_size, 3), dtype=np.uint8
        )
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
        fontScale = 1
        color = (255, 255, 0)
        thickness = 2

        cv2.putText(
            virtual_board,
            str(self._completed_lines),
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.imshow("Virtual Board", virtual_board)

    def update(self, board: Board, current_piece: Piece, next_piece: Piece=None):
        self.board = board
        self._last_piece = self.current_piece
        if current_piece is not None:
            self.current_piece = current_piece
        self.next_piece = next_piece

    def diff_state(self, other_state: "GameState") -> bool:
        other_state.board.board[other_state.board.board > 1] = 1
        if np.array_equal(self.board.board, other_state.board.board):
            print("States did match")
            return True
        else:
            print("States do not match")
            return False

    def clone(self):
        return copy.deepcopy(self)

    def new_piece(self) -> bool:
        if self.current_piece:
            _, cp_y = self.current_piece.zero_based_corner_xy
        else:
            cp_y = 0
        if self._last_piece:
            _, lp_y = self._last_piece.zero_based_corner_xy
        else:
            lp_y = 0
        return (
            self.current_piece
            and self._last_piece is None
            or (self.current_piece and self._last_piece and cp_y < lp_y)
        )

    def move_down(self):
        if self.move_down_possible():
            self.current_piece.move_down()
            return True
        else:
            self.place_piece_on_board()
            return False

    def move_left(self):
        if self.move_left_possible():
            self.current_piece.move_left()
            return True
        return False

    def move_right(self) -> bool:
        if self.move_right_possible():
            self.current_piece.move_right()
            return True
        return False

    def move_down_possible(self) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        py = py + 1
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0 and py + y >= 0:
                if (
                    py + y >= self.board.rows
                    or px + x >= self.board.columns
                    or self.board.board[py + y, px + x] != 0
                ):
                    return False
        return True

    def move_left_possible(self) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        px = px - 1
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0 and py + y >= 0:
                if px + x < 0 or self.board.board[py + y, px + x] != 0:
                    return False
        return True

    def move_right_possible(self) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        px = px + 1
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0 and px + x >= 0 and py + y >= 0:
                if (
                    px + x >= self.board.columns
                    or self.board.board[py + y, px + x] != 0
                ):
                    return False
        return True

    def rot_ccw_possible(self):
        p = self.current_piece.clone()
        p.rot_ccw()
        return self.rot_possible(p.shape)

    def rot_cw_possible(self):
        p = self.current_piece.clone()
        p.rot_cw()
        return self.rot_possible(p.shape)

    def rot_possible(self, shape) -> bool:
        px, py = self.current_piece.zero_based_corner_xy
        for (y, x), value in np.ndenumerate(shape):
            if value != 0 and py + y >= 0:
                if (
                    py + y >= self.board.rows
                    or px + x < 0
                    or px + x >= self.board.columns
                    or self.board.board[py + y, px + x] != 0
                ):
                    return False
        return True

    def rot_ccw(self) -> bool:
        if self.rot_ccw_possible():
            self.current_piece.rot_ccw()
            return True
        return False

    def rot_cw(self):
        if self.rot_cw_possible():
            self.current_piece.rot_cw()
            return True
        return False

    def place_piece_on_board(self):
        px, py = self.current_piece.zero_based_corner_xy
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0:
                self.board.board[
                    py + y,
                    px + x,
                ] = value
        self.board.updated()

    def check_game_over(self):
        return np.any(self.board.board[0] != 0)

    def check_full_lines(self) -> int:
        full_lines = np.count_nonzero(np.all(self.board.board != 0, 1))
        if full_lines != 0:
            removed_lines = self.board.board[np.all(self.board.board != 0, 1)]
            removed_lines.fill(0)
            clean_board = self.board.board[np.any(self.board.board == 0, 1)]
            self.board.board = np.vstack((removed_lines, clean_board))
        self._completed_lines += full_lines
        return full_lines

    def count_holes(self) -> int:
        return self.board.count_holes()

    def cumulative_height(self) -> int:
        return self.board.cumulative_height()

    def roughness(self) -> int:
        return self.board.roughness()

    def relative_height(self) -> int:
        return self.board.relative_height()

    def absolute_height(self) -> int:
        return self.board.height()

    def well_count(self) -> int:
        return self.board.well_count()
