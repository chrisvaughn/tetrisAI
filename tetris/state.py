import copy
from typing import Union

import numpy as np

from .board import Board
from .pieces import Piece


class GameState:
    def __init__(self, board: Board, current_piece: Piece, next_piece: Piece):
        self.board = board
        self.current_piece = current_piece
        self.next_piece = next_piece
        self._last_piece: Union[Piece, None] = None

    def update(self, board: Board, current_piece: Piece, next_piece: Piece):
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
        return (
            self.current_piece
            and self._last_piece is None
            or (
                self.current_piece
                and self._last_piece
                and self.current_piece.y < self._last_piece.y
            )
        )

    def move_down(self):
        if self.move_down_possible():
            self.current_piece.move_down()
        else:
            if self.check_game_over():
                return False
            self.place_piece_on_board()
        return True

    def move_left(self):
        if self.move_left_possible():
            self.current_piece.move_left()

    def move_right(self):
        if self.move_right_possible():
            self.current_piece.move_right()

    def move_down_possible(self):
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            tx = self.current_piece.x
            ty = self.current_piece.y + 1
            if value != 0 and ty + y >= 0:
                if ty + y >= self.board.rows or self.board.board[ty + y, tx + x] != 0:
                    return False
        return True

    def move_left_possible(self):
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            tx = self.current_piece.x - 1
            ty = self.current_piece.y
            if value != 0 and ty + y >= 0:
                if tx + x < 0 or self.board.board[ty + y, tx + x] != 0:
                    return False
        return True

    def move_right_possible(self):
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            tx = self.current_piece.x + 1
            ty = self.current_piece.y
            if value != 0 and tx + x >= 0 and ty + y >= 0:
                if (
                    tx + x >= self.board.columns
                    or self.board.board[ty + y, tx + x] != 0
                ):
                    return False
        return True

    def rot_ccw_possible(self):
        return self.rot_possible(np.rot90(self.current_piece.shape))

    def rot_possible(self, shape):
        for (y, x), value in np.ndenumerate(shape):
            tx = self.current_piece.x
            ty = self.current_piece.y
            if value != 0 and ty + y >= 0:
                if (
                    ty + y >= self.board.rows
                    or tx + x < 0
                    or tx + x >= self.board.columns
                    or self.board.board[ty + y, tx + x] != 0
                ):
                    return False
        return True

    def rotate_ccw(self):
        if self.rot_ccw_possible():
            self.current_piece.rotate_ccw()

    def place_piece_on_board(self):
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0:
                self.board.board[
                    self.current_piece.y + y, self.current_piece.x + x
                ] = value
        self.board.updated()

    def check_game_over(self):
        for (y, x), value in np.ndenumerate(self.current_piece.shape):
            if value != 0 and self.current_piece.y + y < 0:
                return True
        return False

    def check_full_lines(self) -> int:
        full_lines = np.count_nonzero(np.all(self.board.board != 0, 1))
        if full_lines != 0:
            removed_lines = self.board.board[np.all(self.board.board != 0, 1)]
            removed_lines.fill(0)
            clean_board = self.board.board[np.any(self.board.board == 0, 1)]
            self.board.board = np.vstack((removed_lines, clean_board))
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
