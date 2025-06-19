from abc import ABC, abstractmethod
from itertools import zip_longest
from typing import List, Tuple, Union, Optional
import numpy as np

from tetris import GameState, Board, Piece


class BotMove:
    """Represents a move that a bot wants to make"""
    
    def __init__(self, rotations: int, translation: int, score: float = 0.0, end_state=None, lines_completed: int = 0):
        self.rotations = rotations
        self.translation = translation
        self.score = score
        self.end_state = end_state  # Expected game state after executing this move
        self.lines_completed = lines_completed  # Number of lines completed by this move
    
    def to_sequence(self) -> List[Tuple[str]]:
        """Convert the move to a sequence of game actions"""
        rotations = []
        translations = []

        if self.rotations == 3:
            rotations.append("rot_ccw")
        else:
            for i in range(self.rotations):
                rotations.append("rot_cw")
        
        if self.translation < 0:
            for _ in range(abs(self.translation)):
                translations.append("move_left")
        if self.translation > 0:
            for _ in range(abs(self.translation)):
                translations.append("move_right")
        
        seq = list(zip_longest(rotations, translations, fillvalue="noop"))
        if not seq:
            seq.append(("noop",))
        return seq


class BaseBot(ABC):
    """
    Abstract base class for all Tetris bots.
    
    This class defines the interface that all bot implementations must follow.
    Subclasses should implement the specific logic for move evaluation and selection.
    """
    
    def __init__(self, name: str = "BaseBot"):
        self.name = name
        self._current_state: Optional[GameState] = None
        self._detection_count: int = 0
    
    @property
    def current_state(self) -> Optional[GameState]:
        """Get the current game state"""
        return self._current_state
    
    def update_state(self, state: GameState):
        """Update the bot's knowledge of the current game state"""
        self._current_state = state
        self._detection_count += 1
    
    def update_from_detection(self, board: Board, current_piece: Piece, next_piece: Optional[Piece] = None):
        """Update the bot's state from detected board and piece information"""
        if self._current_state is None:
            self._current_state = GameState()
        self._current_state.update(board, current_piece, next_piece)
        self._detection_count += 1
    
    @abstractmethod
    def evaluate_move(self, rotations: int, translation: int) -> float:
        """
        Evaluate a specific move and return its score.
        
        Args:
            rotations: Number of clockwise rotations (0-3)
            translation: Horizontal translation (negative = left, positive = right)
            
        Returns:
            Score for this move (higher is better)
        """
        pass
    
    @abstractmethod
    def get_best_move(self, debug: bool = False) -> Tuple[BotMove, float, int]:
        """
        Find the best move for the current game state.
        
        Args:
            debug: Whether to print debug information
            
        Returns:
            Tuple of (best_move, time_taken, moves_considered)
        """
        pass
    
    def get_move_sequence(self, debug: bool = False) -> List[Tuple[str]]:
        """
        Get the sequence of actions to execute the best move.
        
        Args:
            debug: Whether to print debug information
            
        Returns:
            List of action tuples to execute
        """
        best_move, _, _ = self.get_best_move(debug)
        return best_move.to_sequence()
    
    def reset(self):
        """Reset the bot's internal state"""
        self._current_state = None
        self._detection_count = 0
    
    def get_stats(self) -> dict:
        """Get statistics about the bot's performance"""
        return {
            "name": self.name,
            "detection_count": self._detection_count,
            "current_state": self._current_state is not None
        } 