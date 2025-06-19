import random
import time
from typing import List, Tuple

from tetris import GameState, Board, Piece, InvalidMove

from .base import BaseBot, BotMove


class RandomBot(BaseBot):
    """
    A simple bot that makes random moves.
    
    This demonstrates how to implement a basic bot using the BaseBot abstraction.
    """
    
    def __init__(self, name: str = "RandomBot"):
        super().__init__(name)
        self.random_seed = None
    
    def set_seed(self, seed: int):
        """Set the random seed for reproducible behavior"""
        self.random_seed = seed
        random.seed(seed)
    
    def evaluate_move(self, rotations: int, translation: int) -> float:
        """Evaluate a move by returning a random score"""
        if self._current_state is None:
            raise ValueError("No current state available")
        
        # Check if the move is valid
        state = self._current_state.clone()
        try:
            # Try to execute the move
            state.rot_cw(rotations)
            if translation < 0:
                state.move_left(abs(translation))
            if translation > 0:
                state.move_right(translation)
            
            # Try to drop the piece
            while state.move_down_possible():
                state.move_down()
            state.move_down()
            
            # If we get here, the move is valid
            return random.random()
        except InvalidMove:
            return float('-inf')  # Invalid moves get the worst possible score
    
    def get_best_move(self, debug: bool = False) -> Tuple[BotMove, float, int]:
        """Find a random valid move"""
        if self._current_state is None:
            raise ValueError("No current state available")
        
        start_time = time.time()
        
        # Get all possible moves
        possible_moves = []
        meaningful_rotations = len(self._current_state.current_piece.shapes)
        
        for rot in range(meaningful_rotations):
            piece = self._current_state.current_piece.clone()
            for r in range(rot):
                piece.rot_cw()
            
            left_trans, right_trans = piece.possible_translations()
            for trans in range(-left_trans, right_trans + 1):
                score = self.evaluate_move(rot, trans)
                if score > float('-inf'):
                    # Simulate the move to get the end state
                    end_state = self._simulate_move(rot, trans)
                    possible_moves.append((rot, trans, score, end_state))
        
        if not possible_moves:
            # If no valid moves, return a no-op move
            best_move = BotMove(0, 0, 0.0, self._current_state.clone(), 0)
            time_taken = time.time() - start_time
            return best_move, time_taken, 0
        
        # Select a random move from valid moves
        rot, trans, score, end_state = random.choice(possible_moves)
        # Calculate lines completed
        lines_completed = 0
        if end_state:
            lines_completed = end_state.check_full_lines()
        best_move = BotMove(rot, trans, score, end_state, lines_completed)
        time_taken = time.time() - start_time
        
        if debug:
            print(f"RandomBot selected move: rotations={rot}, translation={trans}, score={score}")
        
        return best_move, time_taken, len(possible_moves)
    
    def _simulate_move(self, rotations: int, translation: int):
        """Simulate a move and return the resulting game state"""
        if self._current_state is None:
            return None
        
        state = self._current_state.clone()
        try:
            # Execute the move
            state.rot_cw(rotations)
            if translation < 0:
                state.move_left(abs(translation))
            if translation > 0:
                state.move_right(translation)
            
            # Drop the piece
            while state.move_down_possible():
                state.move_down()
            state.move_down()
            
            return state
        except InvalidMove:
            return None
    
    def get_stats(self) -> dict:
        """Get statistics about the bot's performance"""
        stats = super().get_stats()
        stats.update({
            "random_seed": self.random_seed,
            "bot_type": "random"
        })
        return stats 