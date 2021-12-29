import threading
import time

from .board import Board
from .state import GameState


class Game:
    def __init__(self, seed):
        if seed:
            self.state = GameState(seed)
        else:
            self.state = GameState(int(time.time()))
        self.state_lock = threading.Lock()
        self.gravity = 10
        self.game_over = False
        self.lines = 0
        self.piece_count = 0
        self.game_thread = threading.Thread(target=self.run)
        self.state.board = Board()

    def start(self):
        self.game_thread.start()

    def display(self):
        self.state.display()

    def run(self) -> int:
        cp = self.state.select_next_piece()
        self.state_lock.acquire()
        self.state.update(self.state.board, cp)
        self.state_lock.release()
        self.piece_count += 1
        time.sleep(1)
        while not self.game_over:
            time.sleep(1 / self.gravity)
            self.state_lock.acquire()
            moved_down = self.state.move_down()
            self.state.update(self.state.board, cp)
            self.state_lock.release()
            if not moved_down:
                self.game_over = self.state.check_game_over()
                if self.game_over:
                    return self.lines
                else:
                    self.lines += self.state.check_full_lines()
                    cp = self.state.select_next_piece()
                    self.state_lock.acquire()
                    self.state.update(self.state.board, cp)
                    self.state_lock.release()
                    self.piece_count += 1

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
