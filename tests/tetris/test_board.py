import numpy as np

from tetris import Board


def test_board_init():
    b = Board()
    assert b.count_holes() == 0
    assert b.cumulative_height() == 0
    assert b.roughness() == 0
    assert b.relative_height() == 0
    assert not b.game_over()


def test_full_row():
    board = np.zeros((Board.rows, Board.columns), dtype=int)
    row_to_fill = 12
    board[row_to_fill] = [1] * 10
    board[row_to_fill - 1] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    b = Board(board)
    assert b.count_holes() == (Board.rows - row_to_fill - 1) * Board.columns
    assert b.cumulative_height() == ((Board.rows - row_to_fill) * Board.columns) + 2
    assert b.roughness() == 3
    assert b.relative_height() == 1
    assert not b.game_over()
