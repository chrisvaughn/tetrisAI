import time
from typing import List, Tuple, Optional

from tetris import GameState, Board, Piece, InvalidMove

from .base import BaseBot, BotMove
from .evaluate import Evaluator, Weights, execute_move


class WeightedBot(BaseBot):
    """
    A bot that uses weighted evaluation functions to make decisions.
    
    This is a wrapper around the existing Evaluator-based bot implementation.
    """
    
    def __init__(self, weights: Weights, parallel: bool = True, scoring: str = "v1", name: str = "WeightedBot"):
        super().__init__(name)
        self.weights = weights
        self.parallel = parallel
        self.scoring = scoring
        self._evaluator: Optional[Evaluator] = None
    
    def update_state(self, state: GameState):
        """Update the bot's knowledge of the current game state"""
        super().update_state(state)
        if self._evaluator is None:
            self._evaluator = Evaluator(state, self.weights, self.parallel, self.scoring)
        else:
            self._evaluator.update_state(state)
    
    def update_from_detection(self, board: Board, current_piece: Piece, next_piece: Optional[Piece] = None):
        """Update the bot's state from detected board and piece information"""
        super().update_from_detection(board, current_piece, next_piece)
        if self._evaluator is None:
            self._evaluator = Evaluator(self._current_state, self.weights, self.parallel, self.scoring)
        else:
            self._evaluator.update_state(self._current_state)
    
    def update_weights(self, weights: Weights):
        """Update the evaluation weights"""
        self.weights = weights
        if self._evaluator is not None:
            self._evaluator.update_weights(weights)
    
    def evaluate_move(self, rotations: int, translation: int) -> float:
        """Evaluate a specific move and return its score"""
        if self._current_state is None:
            raise ValueError("No current state available")
        
        state = self._current_state.clone()
        try:
            execute_move(state, rotations, translation)
        except InvalidMove:
            return float('-inf')  # Invalid moves get the worst possible score
        
        score, _ = self._evaluator.scoring_func(state)
        return score
    
    def get_best_move(self, debug: bool = False) -> Tuple[BotMove, float, int]:
        """Find the best move for the current game state"""
        if self._evaluator is None:
            raise ValueError("No evaluator available - call update_state first")
        
        start_time = time.time()
        best_move, time_taken, moves_considered = self._evaluator.best_move(debug)
        
        # Convert the Evaluator's Move to our BotMove
        bot_move = BotMove(
            rotations=best_move.rotations,
            translation=best_move.translation,
            score=best_move.score,
            end_state=best_move.end_state,  # Include the expected end state
            lines_completed=best_move.lines_completed  # Include lines completed
        )
        
        return bot_move, time_taken, moves_considered
    
    def reset(self):
        """Reset the bot's internal state"""
        super().reset()
        self._evaluator = None
    
    def get_stats(self) -> dict:
        """Get statistics about the bot's performance"""
        stats = super().get_stats()
        stats.update({
            "weights": self.weights,
            "parallel": self.parallel,
            "scoring": self.scoring,
            "evaluator_available": self._evaluator is not None
        })
        return stats 