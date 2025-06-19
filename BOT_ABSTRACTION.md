# Tetris Bot Abstraction

This document describes the new bot abstraction system that allows you to create multiple types of bots with a unified interface.

## Overview

The bot abstraction provides a common interface (`BaseBot`) that all bot implementations must follow. This allows you to easily switch between different bot types and create new bot implementations without changing the game logic.

## Architecture

### Core Classes

#### `BaseBot` (Abstract Base Class)

- **Location**: `bot/base.py`
- **Purpose**: Defines the interface that all bots must implement
- **Key Methods**:
  - `update_state(state)`: Update bot's knowledge of game state
  - `evaluate_move(rotations, translation)`: Evaluate a specific move
  - `get_best_move(debug)`: Find the best move for current state
  - `get_move_sequence(debug)`: Get action sequence for best move

#### `BotMove`

- **Location**: `bot/base.py`
- **Purpose**: Represents a move that a bot wants to make
- **Properties**:
  - `rotations`: Number of clockwise rotations (0-3)
  - `translation`: Horizontal translation (negative = left, positive = right)
  - `score`: Evaluation score for this move

### Concrete Implementations

#### `WeightedBot`

- **Location**: `bot/weighted_bot/weighted_bot.py`
- **Purpose**: Wrapper around the existing Evaluator-based bot
- **Features**:
  - Uses weighted evaluation functions
  - Supports parallel processing
  - Multiple scoring algorithms (v1, v2)
  - Configurable weights

#### `RandomBot`

- **Location**: `bot/random_bot/random_bot.py`
- **Purpose**: Simple bot that makes random moves
- **Features**:
  - Demonstrates the abstraction interface
  - Useful for testing and baseline comparison
  - Configurable random seed

## Usage Examples

### Basic Usage

```python
from bot import WeightedBot, RandomBot, by_mode
from tetris import Game

# Create a weighted bot
weights = by_mode["lines"]
bot = WeightedBot(weights, name="MyBot")

# Create a game
game = Game(seed=12345, level=19)
game.start()

# Update bot with current state
bot.update_state(game.state)

# Get the best move
best_move, time_taken, moves_considered = bot.get_best_move()
print(f"Best move: {best_move.rotations} rotations, {best_move.translation} translation")

# Get the action sequence
actions = best_move.to_sequence()
print(f"Actions: {actions}")
```

### Creating a Custom Bot

```python
from bot.base import BaseBot, BotMove
from tetris import GameState, InvalidMove

class MyCustomBot(BaseBot):
    def __init__(self, name: str = "MyCustomBot"):
        super().__init__(name)
        # Add your custom initialization here
    
    def evaluate_move(self, rotations: int, translation: int) -> float:
        """Implement your custom move evaluation logic"""
        if self._current_state is None:
            raise ValueError("No current state available")
        
        # Your custom evaluation logic here
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
            
            # Calculate your custom score
            score = self._my_custom_scoring(state)
            return score
        except InvalidMove:
            return float('-inf')
    
    def get_best_move(self, debug: bool = False):
        """Implement your custom move selection logic"""
        # Your custom logic to find the best move
        # This is a simple example - you can make it as complex as needed
        best_score = float('-inf')
        best_move = BotMove(0, 0, 0.0)
        
        # Try all possible moves
        for rot in range(4):
            for trans in range(-5, 6):  # Reasonable translation range
                score = self.evaluate_move(rot, trans)
                if score > best_score:
                    best_score = score
                    best_move = BotMove(rot, trans, score)
        
        return best_move, 0.0, 40  # time_taken, moves_considered
    
    def _my_custom_scoring(self, state):
        """Your custom scoring function"""
        # Example: prefer fewer holes
        holes, _ = state.count_holes()
        return -holes  # Negative because fewer holes is better
```

### Running Games with Different Bots

```python
from bot import WeightedBot, RandomBot, by_mode
from tetris import Game

def run_game_with_bot(bot, seed=12345, max_moves=100):
    game = Game(seed=seed, level=19)
    game.start()
    
    move_count = 0
    while not game.game_over and move_count < max_moves:
        bot.update_state(game.state)
        
        if game.state.new_piece():
            move_count += 1
            best_move, _, _ = bot.get_best_move()
            actions = best_move.to_sequence()
            
            # Execute actions
            for action_tuple in actions:
                for action in action_tuple:
                    if action != "noop":
                        getattr(game, action)()
            game.move_seq_complete()
        else:
            game.move_down()
    
    return game.lines, game.piece_count

# Compare different bots
bots = [
    RandomBot("Random"),
    WeightedBot(by_mode["lines"], name="Lines"),
    WeightedBot(by_mode["score"], name="Score"),
]

for bot in bots:
    lines, pieces = run_game_with_bot(bot)
    print(f"{bot.name}: {lines} lines, {pieces} pieces")
```

## Integration with Existing Code

The new abstraction is designed to be backward compatible. You can gradually migrate from the old `Evaluator` class to the new `WeightedBot`:

### Old Way

```python
from bot import Evaluator, by_mode
evaluator = Evaluator(game.state, by_mode["lines"])
best_move, time_taken, moves_considered = evaluator.best_move()
```

### New Way

```python
from bot import WeightedBot, by_mode
bot = WeightedBot(by_mode["lines"])
bot.update_state(game.state)
best_move, time_taken, moves_considered = bot.get_best_move()
```

## Benefits

1. **Unified Interface**: All bots use the same interface, making them interchangeable
2. **Easy Testing**: You can easily compare different bot implementations
3. **Extensibility**: New bot types can be added without changing existing code
4. **Modularity**: Bot logic is separated from game logic
5. **Reusability**: Bots can be used in different contexts (in-memory games, emulator games, etc.)

## Future Bot Types

With this abstraction, you can easily implement various bot types:

- **Neural Network Bot**: Uses a trained neural network for move evaluation
- **Monte Carlo Bot**: Uses Monte Carlo tree search
- **Genetic Algorithm Bot**: Uses evolved strategies
- **Rule-based Bot**: Uses hardcoded rules and heuristics
- **Hybrid Bot**: Combines multiple approaches

## Example Script

Run the example script to see the abstraction in action:

```bash
python example_bot_usage.py
```

This will demonstrate the interface features and compare different bot implementations.
