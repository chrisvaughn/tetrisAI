import copy

import numpy as np

from .board import Board
from .pieces import Piece


class GameState:
    def __init__(self, board: Board, current_piece: Piece, next_piece: Piece):
        self._board = board
        self._current_piece = current_piece
        self._next_piece = next_piece
        self._last_current_piece = None
        self._new_piece_evaluated = False

    def update(self, board: Board, current_piece: Piece, next_piece: Piece):
        self._board = board
        if current_piece is not None:
            self._current_piece = current_piece
        self._next_piece = next_piece

    def new_piece(self) -> bool:
        if self._current_piece is not None and self._last_current_piece is None:
            self._last_current_piece = self._current_piece
            return True
        elif not np.array_equal(
            self._current_piece.shape, self._last_current_piece.shape
        ):
            self._last_current_piece = self._current_piece
            return True
        return False

    def clone(self):
        return copy.deepcopy(self)

    def move_down(self):
        if self.move_down_possible():
            self._current_piece.move_down()
        else:
            if self.check_game_over():
                return False
            self.place_piece_on_board()
        return True

    def move_left(self):
        if self.move_left_possible():
            self._current_piece.move_left()

    def move_right(self):
        if self.move_right_possible():
            self._current_piece.move_right()

    def move_down_possible(self):
        for (y, x), value in np.ndenumerate(self._current_piece.shape):
            tx = self._current_piece.x
            ty = self._current_piece.y + 1
            if value != 0 and ty + y >= 0:
                if ty + y >= self._board.rows or self._board.board[ty + y, tx + x] != 0:
                    return False
        return True

    def move_left_possible(self):
        for (y, x), value in np.ndenumerate(self._current_piece.shape):
            tx = self._current_piece.x - 1
            ty = self._current_piece.y
            if value != 0 and ty + y >= 0:
                if tx + x < 0 or self._board.board[ty + y, tx + x] != 0:
                    return False
        return True

    def move_right_possible(self):
        for (y, x), value in np.ndenumerate(self._current_piece.shape):
            tx = self._current_piece.x + 1
            ty = self._current_piece.y
            if value != 0 and tx + x >= 0 and ty + y >= 0:
                if (
                    tx + x >= self._board.columns
                    or self._board.board[ty + y, tx + x] != 0
                ):
                    return False
        return True

    def rot_right_possible(self):
        return self.rot_possible(np.rot90(self._current_piece.shape))

    def rot_left_possible(self):
        return self.rot_possible(np.rot90(self._current_piece.shape, 3))

    def rot_possible(self, shape):
        for (y, x), value in np.ndenumerate(shape):
            tx = self._current_piece.x
            ty = self._current_piece.y
            if value != 0 and ty + y >= 0:
                if (
                    ty + y >= self._board.rows
                    or tx + x < 0
                    or tx + x >= self._board.columns
                    or self._board.board[ty + y, tx + x] != 0
                ):
                    return False
        return True

    def rotate_left(self):
        if self.rot_left_possible():
            self._current_piece.rotate_left()

    def rotate_right(self):
        if self.rot_right_possible():
            self._current_piece.rotate_right()

    def place_piece_on_board(self):
        for (y, x), value in np.ndenumerate(self._current_piece.shape):
            if value != 0:
                self._board.board[
                    self._current_piece.y + y, self._current_piece.x + x
                ] = value

    def check_game_over(self):
        for (y, x), value in np.ndenumerate(self._current_piece.shape):
            if value != 0 and self._current_piece.y + y < 0:
                return True
        return False

    def check_full_lines(self) -> int:
        full_lines = np.count_nonzero(np.all(self._board.board != 0, 1))
        if full_lines != 0:
            removed_lines = self._board.board[np.all(self._board.board != 0, 1)]
            removed_lines.fill(0)
            clean_board = self._board.board[np.any(self._board.board == 0, 1)]
            self._board.board = np.vstack((removed_lines, clean_board))
        return full_lines

    def count_holes(self) -> int:
        return self._board.count_holes()

    def cumulative_height(self) -> int:
        return self._board.cumulative_height()

    def roughness(self) -> int:
        return self._board.roughness()
