import numpy as np

from tetris import Board


def test_board_init():
    b = Board()
    assert b.count_holes() == (0, 0)
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
    assert b.count_holes() == (
        (Board.rows - row_to_fill - 1) * Board.columns,
        sum([Board.columns * i for i in range(Board.rows - 1, row_to_fill, -1)]),
    )
    assert b.cumulative_height() == ((Board.rows - row_to_fill) * Board.columns) + 2
    assert b.roughness() == 3
    assert b.relative_height() == 1
    assert not b.game_over()


def test_well_count():
    board = np.zeros((Board.rows, Board.columns), dtype=int)
    board[10:, 1] = 1
    board[15:, 3] = 1
    board[15:, 4] = 1
    board[13:, 6] = 1
    b = Board(board)
    assert b.deep_well_count() == 3


def test_unreachable_cells_empty_board():
    b = Board()
    assert b.count_unreachable_cells() == 0


def test_unreachable_cells_no_cavity():
    board = np.zeros((Board.rows, Board.columns), dtype=int)
    # Deep well in column 5 — open to the top, reachable
    board[15:, :5] = 1
    board[15:, 6:] = 1
    b = Board(board)
    assert b.count_unreachable_cells() == 0


def test_unreachable_cells_sealed_cavity():
    board = np.zeros((Board.rows, Board.columns), dtype=int)
    # Build a sealed box: top/bottom/left/right walls enclosing row 11, cols 3-6
    board[10, 2:8] = 1  # top seal
    board[12, 2:8] = 1  # bottom seal
    board[10:13, 2] = 1  # left seal
    board[10:13, 7] = 1  # right seal
    b = Board(board)
    # Cells (11,3), (11,4), (11,5), (11,6) are sealed off from the top
    assert b.count_unreachable_cells() == 4


def test_unreachable_cells_partial_seal():
    board = np.zeros((Board.rows, Board.columns), dtype=int)
    # Overhang that covers most of row 5 but leaves col 9 open
    board[5, :9] = 1
    b = Board(board)
    # Everything below row 5 is still reachable via col 9
    assert b.count_unreachable_cells() == 0
