import random
from dataclasses import dataclass
from typing import List

from tetris import GameState


def scoring_v1(state: GameState) -> float:
    a = -0.3
    b = -0.6
    c = 2
    d = -1
    e = -1
    score = (
        a * state.count_holes()
        + b * state.roughness()
        + c * state.check_full_lines()
        + d * int(state.check_game_over())
        + e * state.cumulative_height()
    )
    return score


@dataclass
class Move:
    rotations: int
    translation: int
    score: float


def move_to_sequence(m: Move) -> List[str]:
    seq = []
    for i in range(m.rotations):
        seq.append("rot_left")
    if m.translation < 0:
        for _ in range(abs(m.translation)):
            seq.append("move_left")
    if m.translation > 0:
        for _ in range(abs(m.translation)):
            seq.append("move_right")
    return seq


class Evaluator:
    def __init__(self, state: GameState):
        self._initial_state = state

    def evaluate_all_moves(self) -> List[Move]:
        possible_moves: List[Move] = []
        for rot in range(4):
            for t in range(-5, 5):
                state = self._initial_state.clone()
                for _ in range(rot):
                    state.rotate_right()
                if t < 0:
                    for _ in range(abs(t)):
                        state.move_left()
                if t > 0:
                    for _ in range(t):
                        state.move_right()
                while state.move_down_possible():
                    state.move_down()
                state.move_down()
                score = scoring_v1(state)
                possible_moves.append(Move(rot, t, score))
        return possible_moves

    def best_move_sequence(self) -> List[str]:
        all_moves = sorted(
            self.evaluate_all_moves(), key=lambda x: x.score, reverse=True
        )
        print(all_moves)
        best_move = all_moves[0]
        print(best_move)
        return move_to_sequence(best_move)

    def random_valid_move_sequence(self) -> List[str]:
        sequence: List[str] = []
        rot_left = random.random() > 0.5
        for i in range(random.randint(0, 3)):
            if rot_left:
                if self._initial_state.rot_left_possible():
                    self._initial_state.rotate_left()
                    sequence.append("rot_left")
            else:
                if self._initial_state.rot_right_possible():
                    self._initial_state.rotate_right()
                    sequence.append("rot_right")

        left = random.random() > 0.5
        for i in range(random.randint(0, 5)):
            if left:
                if self._initial_state.move_left_possible():
                    self._initial_state.move_left()
                    sequence.append("move_left")
            else:
                if self._initial_state.move_right_possible():
                    self._initial_state.move_right()
                    sequence.append("move_right")

        while self._initial_state.move_down_possible():
            self._initial_state.move_down()
            sequence.append("move_down")

        return sequence
