import threading
import time
from collections import Counter

from .board import Board
from .state import GameState

# frames per cell in original NES Tetris
frames_per_cell_by_level = {
    0: 48,
    1: 43,
    2: 38,
    3: 33,
    4: 28,
    5: 23,
    6: 18,
    7: 13,
    8: 8,
    9: 6,
    10: 5,
    11: 5,
    12: 5,
    13: 4,
    14: 4,
    15: 4,
    16: 3,
    17: 3,
    18: 3,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
    23: 2,
    24: 2,
    25: 2,
    26: 2,
    27: 2,
    28: 2,
    29: 1,
}

score_by_number_of_lines_cleared = [40, 100, 300, 1200]


class Game:
    def __init__(self, seed=int(time.time() * 1000), level=19):
        self.state = GameState(seed)
        self.state_lock = threading.Lock()
        self.frames_per_cell = frames_per_cell_by_level[level]
        self.framerate = 1 / 60
        self.game_over = False
        self.lines = 0
        self.piece_count = 0
        self.game_thread = threading.Thread(target=self.run)
        self.state.board = Board()
        self.score = 0
        self.piece_stats = Counter()

    def start(self):
        self.game_thread.start()

    def display(self):
        self.state.display()

    def run(self) -> int:
        cp = self.state.select_next_piece()
        self.piece_stats[cp.name] += 1
        self.state_lock.acquire()
        self.state.update(self.state.board, cp)
        self.state_lock.release()
        self.piece_count += 1
        time.sleep(1)
        while not self.game_over:
            frame_start = time.time()
            self.state_lock.acquire()
            moved_down = self.state.move_down()
            self.state.update(self.state.board, cp)
            self.state_lock.release()
            if not moved_down:
                self.game_over = self.state.check_game_over()
                if self.game_over:
                    return self.lines
                else:
                    lines = self.state.check_full_lines()
                    if lines > 0:
                        self.lines += lines
                        if lines <= len(score_by_number_of_lines_cleared):
                            self.score += score_by_number_of_lines_cleared[lines - 1]
                    cp = self.state.select_next_piece()
                    self.piece_stats[cp.name] += 1
                    self.state_lock.acquire()
                    self.state.update(self.state.board, cp)
                    self.state_lock.release()
                    self.piece_count += 1
            time.sleep(
                max(
                    0,
                    self.framerate * self.frames_per_cell - (time.time() - frame_start),
                )
            )

    def move_down(self, moves: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.move_down(moves)
        self.state_lock.release()
        return success

    def move_left(self, moves: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.move_left(moves)
        self.state_lock.release()
        return success

    def move_right(self, moves: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.move_right(moves)
        self.state_lock.release()
        return success

    def rot_ccw(self, rot: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.rot_ccw(rot)
        self.state_lock.release()
        return success

    def rot_cw(self, rot: int = 1) -> bool:
        self.state_lock.acquire()
        success = self.state.rot_cw(rot)
        self.state_lock.release()
        return success

    def move_seq_complete(self):
        self.state_lock.acquire()
        self.state.update(self.state.board, self.state.current_piece)
        self.state_lock.release()
