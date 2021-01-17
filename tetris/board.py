import copy
from typing import List

import numpy as np


class Board:
    columns: int = 10
    rows: int = 20

    def __init__(self, board: np.ndarray):
        self.board = board
        self._peaks = None

    def clone(self):
        return copy.deepcopy(self)

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

    def height(self) -> int:
        h = 0
        for row in reversed(self.board):
            if any(row):
                h += 1
            else:
                break
        return h

    def _get_peaks(self) -> List[int]:
        if self._peaks:
            return self._peaks

        peaks = [Board.rows] * Board.columns
        for y in range(Board.rows):
            for x in range(Board.columns):
                if self.board[y][x] != 0 and peaks[x] == Board.rows:
                    peaks[x] = y
        self._peaks = peaks
        return peaks

    def count_holes(self) -> int:
        peaks = self._get_peaks()
        holes = 0
        for x in range(len(peaks)):
            for y in range(Board.rows - 1, peaks[x], -1):
                if self.board[y][x] == 0:
                    holes += 1
        return holes

    def cumulative_height(self) -> int:
        peaks = self._get_peaks()
        height = 0
        for i in peaks:
            height += 20 - i
        return height

    def roughness(self) -> int:
        peaks = self._get_peaks()
        roughness = 0
        for i in range(len(peaks) - 1):
            roughness += abs(peaks[i] - peaks[i + 1])
        return roughness

    def relative_height(self) -> int:
        peaks = self._get_peaks()
        return max(peaks) - min(peaks)
