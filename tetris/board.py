
import numpy as np

# Pre-computed constants for fixed 10×20 board
_COL_BITS = 1 << np.arange(10, dtype=np.int64)  # [1, 2, 4, ..., 512] for bitmask encoding
_COL_MASK = (1 << 10) - 1  # 0b1111111111
_ROWS_COL = np.arange(20)[:, np.newaxis]  # (20,1) row-index array for count_holes/count_cells


class Board:
    columns: int = 10
    rows: int = 20

    def __init__(self, board: np.ndarray = None):
        if board is None:
            board = np.zeros((Board.rows, Board.columns), dtype=int)
        self.board = board
        self._peaks = None
        self._peaks_list = None

    def clone(self):
        new_board = Board.__new__(Board)
        new_board.board = self.board.copy()
        new_board._peaks = None
        new_board._peaks_list = None
        return new_board

    def compare(self, other: "Board") -> bool:
        return np.array_equal(self.board, other.board)

    def print(self):
        print("-" * 12)
        for y in self.board:
            print("|", end="")
            for x in y:
                print(f"{x}", end="")
            print("|")
        print("-" * 12)

    def updated(self):
        self._peaks = None
        self._peaks_list = None

    def height(self) -> int:
        min_peak = min(self._get_peaks_list())
        return Board.rows - min_peak if min_peak < Board.rows else 0

    def _get_peaks(self) -> np.ndarray:
        if self._peaks is not None:
            return self._peaks
        has_content = self.board.any(axis=0)
        self._peaks = np.where(has_content, self.board.argmax(axis=0), Board.rows)
        self._peaks_list = self._peaks.tolist()
        return self._peaks

    def _get_peaks_list(self) -> list:
        if self._peaks_list is not None:
            return self._peaks_list
        self._get_peaks()
        return self._peaks_list

    def count_holes(self) -> (int, int):
        peaks = self._get_peaks()
        below_peak = _ROWS_COL > peaks
        hole_mask = below_peak & (self.board == 0)
        holes = int(hole_mask.sum())
        weighted_holes = int((_ROWS_COL * hole_mask).sum())
        return holes, weighted_holes

    def cumulative_height(self) -> int:
        return Board.rows * Board.columns - sum(self._get_peaks_list())

    def roughness(self) -> int:
        p = self._get_peaks_list()
        total = 0
        for i in range(len(p) - 1):
            total += abs(p[i + 1] - p[i])
        return total

    def relative_height(self) -> int:
        p = self._get_peaks_list()
        return max(p) - min(p)

    def game_over(self) -> bool:
        return self.board.all()

    def deep_well_count(self) -> int:
        """Count columns that are at least 3 rows deeper than both neighbors."""
        p = self._get_peaks_list()
        padded = [0] + p + [0]
        count = 0
        for i in range(len(p) - 1):
            if padded[i] - p[i] <= -3 and padded[i + 2] - p[i] <= -3:
                count += 1
        return count

    def count_well_cells(self) -> int:
        """
        Total depth of all wells: for each column deeper than both neighbors,
        the contribution is min(center - left, center - right) in row units.
        Walls are treated as peak row 0 (full height).
        """
        p = self._get_peaks_list()
        padded = [0] + p + [0]
        total = 0
        for i in range(len(p)):
            depth = min(p[i] - padded[i], p[i] - padded[i + 2])
            if depth > 0:
                total += depth
        return total

    def count_cells(self) -> (int, int):
        peaks = self._get_peaks()
        below_peak = _ROWS_COL > peaks
        cell_mask = below_peak & (self.board == 1)
        cells = int(cell_mask.sum())
        weighted_cells = int((_ROWS_COL * cell_mask).sum())
        return cells, weighted_cells

    def spawn_zone_filled(self) -> int:
        # Rows 0-1, columns 3-6 (0-indexed) cover the piece spawn area (center x=6 1-based).
        return int(self.board[0:2, 3:7].sum())

    def count_row_transitions(self) -> int:
        b = self.board
        # Adjacent-column differences (avoid np.diff overhead for binary board)
        transitions = int(np.count_nonzero(b[:, 1:] != b[:, :-1]))
        transitions += int(Board.rows - b[:, 0].sum())   # left-border empty cells
        transitions += int(Board.rows - b[:, -1].sum())  # right-border empty cells
        return transitions

    def count_unreachable_cells(self) -> int:
        """Count empty cells sealed off from the top by filled cells."""
        # Convert each row to a Python int bitmask: bit c set iff board[r,c] is empty.
        # Row-by-row top-down BFS: seed each row from the row above, then expand
        # horizontally. Python integer bitwise ops are faster than numpy for 10-wide rows.
        empty = self.board == 0
        empty_ints: list[int] = (empty @ _COL_BITS).tolist()

        empty_count = int(empty.sum())
        reachable = empty_ints[0]
        reachable_count = reachable.bit_count()

        for r in range(1, Board.rows):
            em = empty_ints[r]
            seed = reachable & em  # cells in this row accessible from row above
            if seed:
                prev = -1
                while seed != prev:
                    prev = seed
                    seed = (seed | ((seed << 1) & _COL_MASK) | (seed >> 1)) & em
            reachable = seed
            reachable_count += reachable.bit_count()

        return empty_count - reachable_count
