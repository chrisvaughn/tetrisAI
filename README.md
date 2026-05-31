# tetrisAI

A Tetris AI implementing bot strategies using Genetic Algorithms (GA) with weighted evaluation functions.

## Setup

```bash
uv venv 3.14
```

Get dependencies on macOS for running the AI with an emulator:

```bash
uv sync --extra macOS
```

Get dependencies on Linux for training the bot:

```bash
uv sync
```

## Running the Bot

**In-memory simulation (fast, for training/testing):**

```bash
python run.py --bot-model WeightedBotLines
python run.py --bot-model Random --seed 12345 --stats
```

**With emulator (plays actual NES Tetris):**

```bash
python run.py --emulator --bot-model WeightedBotLines --stats
```

**Using saved weights from training:**

```bash
python run.py --bot-model WeightedBotScore --save-file save_score.pkl --save-gen 50
```

**Common flags:**

- `--stats`: Print move statistics during gameplay
- `--drop`: Enable soft drop (pieces fall faster)
- `--seed <int>`: Set RNG seed for reproducibility
- `--level <int>`: Starting level (default: 19)
- `--debug`: Enable detailed debug logging

## Training

**Genetic Algorithm training (WeightedBot):**

```bash
# Train for lines optimization
python train.py --fitness lines --population 100 --generations 100

# Train for score optimization
python train.py --fitness score --population 100 --generations 50 --save-file custom_save.pkl

# Control parallelism
python train.py --parallel-runners 8
python train.py --no-parallel  # Disable parallel evaluation
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/tetris/test_board.py

# Run with coverage
pytest --cov=tetris --cov=bot
```

## Code Formatting and Linting

```bash
# Format code and sort imports
ruff format .

# Lint and auto-fix
ruff check --fix .

# Run both (common workflow)
ruff check --fix . && ruff format .
```

## References

<https://github.com/IdreesInc/TetNet/blob/master/tetnet.js>
<https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/>
<https://github.com/snej0l/tetris/blob/master/tetris.py>

## Top Scores

515 - 2021-11-29
