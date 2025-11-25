# Tetris Neural Network AI: Implementation Plan

## Executive Summary

This document outlines a progressive approach to building a neural network-based Tetris AI using PyTorch, starting with simple architectures and advancing to state-of-the-art deep reinforcement learning methods. The plan leverages your existing game engine and bot infrastructure while introducing modern deep learning techniques that can discover optimal play strategies beyond hand-crafted evaluation functions.

## Why Neural Networks for Tetris?

### Limitations of Genetic Algorithms with Hand-Crafted Features

Your current GA approach has produced strong results, but has inherent limitations:

1. **Feature Engineering Ceiling**: GA optimizes weights for features *you* defined (holes, roughness, height, etc.). If the optimal strategy depends on patterns you didn't encode, GA cannot discover them.

2. **Linear Combination Constraint**: Weighted sum evaluation assumes features combine linearly. Complex interactions between board states and piece sequences may require non-linear reasoning.

3. **Limited Lookahead**: WeightedBot evaluates immediate moves. Neural networks can learn to implicitly "look ahead" multiple pieces through experience.

4. **State Space Exploration**: GA requires you to hypothesize which board characteristics matter. Neural networks can discover unexpected patterns in raw board states.

### Advantages of Neural Network Approaches

1. **Feature Learning**: Networks learn relevant features directly from board states rather than requiring manual feature engineering.

2. **Non-linear Reasoning**: Deep networks can model complex relationships between board configurations and long-term outcomes.

3. **Sequential Decision Making**: Reinforcement learning naturally handles the temporal credit assignment problem (which move led to the eventual game over?).

4. **Continuous Improvement**: Can learn from millions of games, discovering subtle patterns invisible to human intuition.

5. **Transfer Learning**: Once trained, models can potentially adapt to variations (different starting levels, piece distributions, etc.).

## Research Foundation

### Proven Approaches in Literature

**Deep Q-Learning (DQN)** has emerged as the dominant approach for Tetris AI:

- **Stanford CS231n (2016)**: Demonstrated human-level control through deep reinforcement learning for Tetris [[Playing Tetris with Deep Reinforcement Learning]](https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf)

- **Recent Implementations (2024)**: Q-Learning successfully trained models to play Tetris after ~6,000 games (85,000 moves), using simple networks with 2 hidden layers [[How I trained a Neural Network to Play Tetris]](https://timhanewich.medium.com/how-i-trained-a-neural-network-to-play-tetris-using-reinforcement-learning-ecfa529c767a)

- **Multiple Open-Source Successes**: Several PyTorch implementations demonstrate that DQN can achieve strong performance [[vietnh1009/Tetris-deep-Q-learning-pytorch]](https://github.com/vietnh1009/Tetris-deep-Q-learning-pytorch), [[nuno-faria/tetris-ai]](https://github.com/nuno-faria/tetris-ai)

**Key Techniques**:
- Experience replay for sample efficiency
- Epsilon-greedy exploration (ε starts at 1.0, decays to 0)
- Discount factor (γ ≈ 0.95) for future rewards
- Target network for training stability

## Progressive Implementation Plan

We'll build three models in sequence, each adding complexity:

### Phase 1: Supervised Learning Baseline (1-2 weeks)
**Goal**: Learn to mimic your best GA bot

**Phase 2: Deep Q-Network (DQN) (2-4 weeks)
**Goal**: Learn optimal policy through self-play

**Phase 3: Advanced Architectures (4+ weeks)
**Goal**: State-of-the-art performance with modern techniques

---

## Phase 1: Supervised Learning Baseline

### Objective
Create a neural network that mimics your best WeightedBot by learning from its decisions. This validates your infrastructure and provides a baseline before tackling the harder RL problem.

### Why Start Here?
1. **Simpler Problem**: Supervised learning is easier to debug than RL
2. **Data Generation**: Leverages your already-optimized GA weights
3. **Upper Bound**: Shows best-case performance if network perfectly learns the weighted evaluation
4. **Infrastructure Validation**: Tests data pipeline, training loop, and bot integration
5. **Quick Wins**: Can achieve decent performance in days, not weeks

### Architecture: Simple Feedforward Network

```
Input Layer (200 units)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (1 unit, Linear)
```

**Input Representation**:
- Flattened 10×20 board (200 binary values: 0=empty, 1=filled)
- Simple and interpretable
- Network must learn spatial patterns from scratch

**Output**:
- Single value: predicted "goodness" of board state after move
- Evaluate all legal moves, pick highest scoring

**Loss Function**: Mean Squared Error (MSE)
- Target = score from WeightedBot's evaluation function

### Training Process

1. **Data Collection**:
   ```python
   # Run WeightedBotLines for N games
   # For each piece placement decision:
   #   - Store: (board_state_before, rotation, translation, evaluation_score)
   #   - Store: (board_state_after, score)
   # Save to dataset (e.g., 100k examples)
   ```

2. **Training Loop**:
   - Split data: 80% train, 10% validation, 10% test
   - Batch size: 512
   - Optimizer: Adam (lr=0.001)
   - Epochs: 50-100 with early stopping
   - Monitor validation loss

3. **Evaluation**:
   - Play 100 games with trained bot
   - Compare lines cleared and score to WeightedBotLines
   - Expect: 70-90% of WeightedBot performance (some information loss is normal)

### Expected Outcomes
- **Best case**: Network matches or slightly exceeds WeightedBot (may generalize better)
- **Typical case**: Network achieves 80-85% of WeightedBot performance
- **Success metric**: Network clearly plays better than random bot

### Key Files to Create
- `bot/neural_bot/supervised_model.py`: Network architecture
- `bot/neural_bot/supervised_bot.py`: Bot implementation
- `bot/neural_bot/data_collection.py`: Generate training data from WeightedBot
- `bot/neural_bot/train_supervised.py`: Training script

---

## Phase 2: Deep Q-Network (DQN)

### Objective
Train a network to learn optimal policy through self-play using reinforcement learning, without requiring labeled data from human or bot experts.

### Why DQN?

1. **Industry Standard**: Most successful approach for Tetris in research and practice
2. **Well-Understood**: Extensive literature and debugging techniques
3. **Sample Efficient**: Experience replay allows learning from past experiences
4. **Proven Results**: Can exceed human performance with enough training

### Reinforcement Learning Framework

**State (S)**: Current board configuration + current piece + next piece

**Action (A)**: Discrete move = (rotation, translation) pair
- ~68 possible actions per piece (4 rotations × 17 translations)

**Reward (R)**: Design options (experiment with these):
1. **Sparse**: +1 per line cleared, -1 for game over
2. **Dense**: +1 per line, +0.01 per move survived, -1 for game over
3. **Score-based**: Actual NES Tetris score (includes level multiplier)

**Q-Value**: Expected cumulative reward for taking action A in state S

### Architecture: Deep Q-Network

```
Input Layer (224 units)
    ↓
Dense (256 units, ReLU)
    ↓
Dense (256 units, ReLU)
    ↓
Dense (128 units, ReLU)
    ↓
Output Layer (68 units, Linear)  # Q-value for each possible action
```

**Input Representation**:
- Board state: 10×20 = 200 values (binary or normalized)
- Current piece: 7 values (one-hot encoding of piece type)
- Next piece: 7 values (one-hot encoding)
- Rotation: 4 values (one-hot encoding)
- Total: 218 values

**Alternative: Add hand-crafted features**:
- Include your 11 board metrics (holes, height, etc.) as additional inputs
- This "bootstraps" learning with domain knowledge
- Allows network to focus on learning strategy rather than feature extraction

**Output**:
- Q-value for each valid (rotation, translation) combination
- Use mask to prevent invalid actions
- Select action with highest Q-value (exploitation) or random (exploration)

### DQN Algorithm Components

#### 1. Experience Replay
```python
# Store transitions in replay buffer
replay_buffer = deque(maxlen=20000)
replay_buffer.append((state, action, reward, next_state, done))

# Sample random minibatch for training
batch = random.sample(replay_buffer, batch_size=512)
```

**Why it works**:
- Breaks correlation between consecutive samples
- Allows learning from rare events multiple times
- Improves sample efficiency

#### 2. Target Network
```python
# Main network: updated every step
q_network = DQN()

# Target network: updated every N steps (e.g., 1000)
target_network = DQN()
target_network.load_state_dict(q_network.state_dict())
```

**Why it works**:
- Stabilizes training by fixing the target for several updates
- Prevents "chasing a moving target" problem
- Critical for convergence

#### 3. Epsilon-Greedy Exploration
```python
epsilon = 1.0  # Start fully random
epsilon_min = 0.01
epsilon_decay = 0.995

if random.random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(q_network(state))  # Exploit

epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

**Why it works**:
- Early training: explores different strategies
- Late training: exploits learned policy
- Balances exploration vs. exploitation

### Training Process

#### Hyperparameters (starting point)
```python
REPLAY_BUFFER_SIZE = 20000
BATCH_SIZE = 512
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000  # steps
EPISODES = 10000  # ~10k games
```

#### Training Loop
```python
for episode in range(EPISODES):
    state = reset_game()
    episode_reward = 0

    while not game_over:
        # Select action (epsilon-greedy)
        if random() < epsilon:
            action = random_action()
        else:
            with torch.no_grad():
                action = q_network(state).argmax()

        # Execute action in environment
        next_state, reward, done = step(action)

        # Store transition
        replay_buffer.append((state, action, reward, next_state, done))

        # Train on minibatch
        if len(replay_buffer) > BATCH_SIZE:
            batch = sample(replay_buffer, BATCH_SIZE)
            loss = train_step(batch)

        # Update target network
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(q_network.state_dict())

        state = next_state
        episode_reward += reward

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Log progress
    if episode % 100 == 0:
        evaluate_policy()  # Play 10 games without exploration
```

#### Q-Learning Update Rule
```python
# Compute target Q-values
with torch.no_grad():
    max_next_q = target_network(next_states).max(dim=1)[0]
    target_q = rewards + GAMMA * max_next_q * (1 - dones)

# Compute current Q-values
current_q = q_network(states).gather(1, actions)

# Loss and optimization
loss = F.mse_loss(current_q, target_q)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Reward Shaping Experiments

Test multiple reward functions to see what works best:

**Option 1: Line-focused**
```python
reward = lines_cleared  # 0, 1, 2, 3, or 4
if game_over:
    reward = -10
```

**Option 2: Survival + Lines**
```python
reward = lines_cleared + 0.01  # Small reward for each step survived
if game_over:
    reward = -5
```

**Option 3: NES Score**
```python
reward = (score_after - score_before) / 1000.0  # Normalize
if game_over:
    reward = -10
```

**Option 4: Height Penalty** (based on GA insights)
```python
reward = lines_cleared - 0.01 * board.absolute_height()
if game_over:
    reward = -10
```

Start with Option 1 or 2 (simpler), then experiment.

### Expected Training Time
- **Episodes needed**: 5,000-10,000 games
- **Wall time** (single CPU): 12-24 hours
- **Wall time** (GPU): 4-8 hours
- **Performance**: Should exceed WeightedBot after 5k-7k episodes

### Evaluation Metrics
Track during training:
- Average episode reward
- Average lines cleared per game
- Average game score
- Max lines cleared
- Epsilon value

Evaluate every 500 episodes:
- Play 50 games without exploration (epsilon=0)
- Compare to WeightedBot baseline

### Key Files to Create
- `bot/neural_bot/dqn_model.py`: Network architecture
- `bot/neural_bot/dqn_bot.py`: Bot implementation
- `bot/neural_bot/replay_buffer.py`: Experience replay
- `bot/neural_bot/train_dqn.py`: Training script with full DQN algorithm

---

## Phase 3: Advanced Architectures

Once DQN is working, explore these improvements:

### 3.1 Convolutional Neural Network (CNN)

**Motivation**: Treat board as 2D image; CNNs excel at spatial pattern recognition

**Architecture**:
```
Input: 10×20×1 board (or 10×20×2 with current piece overlay)
    ↓
Conv2D (32 filters, 3×3, ReLU, padding=1)
    ↓
Conv2D (64 filters, 3×3, ReLU, padding=1)
    ↓
Conv2D (64 filters, 3×3, ReLU, padding=1)
    ↓
Flatten
    ↓
Dense (256, ReLU)
    ↓
Dense (128, ReLU)
    ↓
Output (68 Q-values)
```

**Why it helps**:
- Automatically learns spatial features (holes, wells, contours)
- Parameter sharing makes training more efficient
- Translation invariance: patterns are similar across board columns

**Expected improvement**: 10-20% better sample efficiency, potentially higher final performance

### 3.2 Double DQN (DDQN)

**Motivation**: Standard DQN overestimates Q-values, hurting performance

**Change to Q-learning update**:
```python
# Standard DQN: uses target network for both action selection and evaluation
max_next_q = target_network(next_states).max(dim=1)[0]

# Double DQN: uses main network for action selection, target for evaluation
best_actions = q_network(next_states).argmax(dim=1)
max_next_q = target_network(next_states).gather(1, best_actions)
```

**Why it helps**: Reduces overestimation bias, more stable learning

**Expected improvement**: 5-15% better performance with same training time

### 3.3 Dueling DQN

**Motivation**: Separate value of being in a state from value of actions

**Architecture**:
```
Input → Shared Layers → Split into two streams:
    ├─ Value Stream: Dense → V(s) [scalar]
    └─ Advantage Stream: Dense → A(s,a) [68 values]

Output: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

**Why it helps**:
- Value stream learns how good the board state is (independent of action)
- Advantage stream learns which action is best
- Faster learning: value doesn't need to relearn for each action

**Expected improvement**: 10-20% better sample efficiency

### 3.4 Prioritized Experience Replay (PER)

**Motivation**: Learn more from surprising/important transitions

**Change**:
```python
# Store with priority based on TD-error
td_error = abs(target_q - current_q)
replay_buffer.append((state, action, reward, next_state, done), priority=td_error)

# Sample with probability proportional to priority
batch = replay_buffer.sample(BATCH_SIZE, beta=0.4)
```

**Why it helps**: Focuses learning on mistakes, accelerates training

**Expected improvement**: 30-50% faster convergence

### 3.5 Multi-Step Returns

**Motivation**: Use rewards from multiple future steps, not just next step

**Change**:
```python
# Standard: R_t + γ * max Q(s_t+1, a)
# 3-step: R_t + γ*R_t+1 + γ²*R_t+2 + γ³ * max Q(s_t+3, a)
```

**Why it helps**: Faster credit assignment for delayed rewards

**Expected improvement**: 20-30% faster convergence

### 3.6 Recurrent Networks (LSTM/GRU)

**Motivation**: Model temporal dependencies and piece sequences

**Architecture**:
```
Input (board + piece)
    ↓
CNN layers (extract spatial features)
    ↓
LSTM (128 hidden units, 2 layers)
    ↓
Dense layers
    ↓
Output (Q-values)
```

**Why it helps**:
- Can learn patterns like "I-piece drought"
- Models dependency between current and next piece
- Implicitly learns to plan ahead

**Expected improvement**: 10-30% better performance, especially at high levels

### 3.7 Rainbow DQN

**Integration**: Combine all improvements above into one model
- Double DQN
- Dueling architecture
- Prioritized replay
- Multi-step returns
- Noisy networks (for exploration)
- Distributional RL (model distribution of returns)

**Expected improvement**: State-of-the-art performance, 2-3x better than basic DQN

---

## Implementation Roadmap

### Weeks 1-2: Phase 1 - Supervised Learning
- [ ] Implement data collection from WeightedBot
- [ ] Create feedforward network architecture
- [ ] Build training pipeline with PyTorch
- [ ] Train and evaluate supervised bot
- [ ] Establish performance baseline

**Deliverables**:
- Working neural bot that mimics GA bot
- Training infrastructure and evaluation scripts
- Baseline performance numbers

### Weeks 3-6: Phase 2 - DQN
- [ ] Implement replay buffer
- [ ] Build DQN architecture (feedforward version)
- [ ] Implement target network and update logic
- [ ] Create epsilon-greedy exploration
- [ ] Implement Q-learning training loop
- [ ] Experiment with reward functions
- [ ] Train for 5,000-10,000 episodes
- [ ] Comprehensive evaluation vs GA baseline

**Deliverables**:
- Working DQN bot trained via self-play
- Reward shaping experiments documented
- Performance comparison with GA bot
- Training curves and analysis

### Weeks 7-10: Phase 3a - CNN and DDQN
- [ ] Implement CNN architecture
- [ ] Train CNN-based DQN
- [ ] Implement Double DQN
- [ ] Compare with feedforward DQN

**Deliverables**:
- CNN-based DQN bot
- Double DQN implementation
- Performance comparison

### Weeks 11-14: Phase 3b - Advanced Techniques
- [ ] Implement Dueling DQN
- [ ] Add prioritized experience replay
- [ ] Experiment with multi-step returns
- [ ] Test LSTM architecture

**Deliverables**:
- Advanced DQN variants
- Ablation studies showing impact of each technique
- Best performing model identified

### Weeks 15+: Phase 3c - Rainbow and Optimization
- [ ] Integrate best techniques into Rainbow DQN
- [ ] Hyperparameter tuning
- [ ] Extended training runs
- [ ] Performance optimization

**Deliverables**:
- State-of-the-art Tetris bot
- Comprehensive benchmarks
- Research writeup

---

## Integration with Existing Codebase

### Directory Structure
```
bot/neural_bot/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── supervised_model.py      # Phase 1
│   ├── dqn_model.py              # Phase 2
│   ├── cnn_dqn_model.py          # Phase 3
│   ├── dueling_dqn_model.py      # Phase 3
│   └── recurrent_dqn_model.py    # Phase 3
├── agents/
│   ├── __init__.py
│   ├── supervised_bot.py         # Inherits from BaseBot
│   ├── dqn_bot.py                # Inherits from BaseBot
│   └── rainbow_bot.py            # Inherits from BaseBot
├── training/
│   ├── __init__.py
│   ├── data_collection.py        # Generate supervised data
│   ├── replay_buffer.py          # Experience replay
│   ├── train_supervised.py       # Phase 1 training
│   ├── train_dqn.py              # Phase 2 training
│   └── train_advanced.py         # Phase 3 training
├── utils/
│   ├── __init__.py
│   ├── state_encoder.py          # Convert GameState to tensors
│   ├── action_decoder.py         # Convert network output to BotMove
│   └── reward_functions.py       # Different reward shaping options
└── configs/
    ├── supervised_config.yaml
    ├── dqn_config.yaml
    └── rainbow_config.yaml
```

### Integrating with BaseBot Interface

Your existing `BaseBot` interface is perfect for neural network bots:

```python
# bot/neural_bot/agents/dqn_bot.py
from bot.base import BaseBot, BotMove
import torch

class DQNBot(BaseBot):
    def __init__(self, model_path, epsilon=0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.epsilon = epsilon  # For evaluation, typically 0
        self.current_state = None

    def update_state(self, state):
        """Receive current game state"""
        self.current_state = state

    def get_best_move(self, debug=False):
        """Return best move using DQN"""
        start_time = time.time()

        # Get all possible moves for current piece
        possible_moves = self._generate_possible_moves()

        if random.random() < self.epsilon:
            # Exploration: random move
            best_move = random.choice(possible_moves)
        else:
            # Exploitation: use network
            best_move = self._select_best_move_with_network(possible_moves)

        time_taken = time.time() - start_time
        moves_considered = len(possible_moves)

        return best_move, time_taken, moves_considered

    def _generate_possible_moves(self):
        """Generate all valid (rotation, translation) pairs"""
        moves = []
        for rotation in range(4):
            left_moves, right_moves = self.current_state.current_piece.possible_translations()
            for translation in range(-left_moves, right_moves + 1):
                # Simulate move and create BotMove object
                test_state = self.current_state.clone()
                # Apply rotation and translation
                for _ in range(rotation):
                    test_state.rot_cw()
                if translation < 0:
                    for _ in range(abs(translation)):
                        test_state.move_left()
                else:
                    for _ in range(translation):
                        test_state.move_right()

                # Check if valid
                if not test_state.check_collision():
                    # Drop to bottom
                    while not test_state.check_collision():
                        test_state.move_down()
                    test_state.move_up()  # Back to valid position
                    test_state.place_piece_on_board()
                    lines = test_state.check_full_lines()

                    moves.append(BotMove(
                        rotations=rotation,
                        translation=translation,
                        score=0,  # Will be set by network
                        end_state=test_state,
                        lines_completed=lines
                    ))
        return moves

    def _select_best_move_with_network(self, possible_moves):
        """Use neural network to evaluate moves"""
        best_move = None
        best_q = -float('inf')

        for move in possible_moves:
            # Encode state after move
            state_tensor = self._encode_state(move.end_state)

            # Get Q-value from network
            with torch.no_grad():
                q_value = self.model(state_tensor).item()

            if q_value > best_q:
                best_q = q_value
                best_move = move

        best_move.score = best_q
        return best_move

    def _encode_state(self, state):
        """Convert GameState to network input tensor"""
        # Board: 10x20 binary matrix
        board_array = (state.board.cells != 0).astype(np.float32)
        board_flat = board_array.flatten()  # 200 values

        # Current piece: one-hot encoding
        piece_encoding = np.zeros(7, dtype=np.float32)
        piece_encoding[state.current_piece.type_index] = 1.0

        # Next piece: one-hot encoding
        next_piece_encoding = np.zeros(7, dtype=np.float32)
        next_piece_encoding[state.next_piece.type_index] = 1.0

        # Concatenate
        input_array = np.concatenate([board_flat, piece_encoding, next_piece_encoding])

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
        return input_tensor.to(self.device)

    def evaluate_move(self, rotations, translation):
        """Required by BaseBot interface"""
        # Create test state
        test_state = self.current_state.clone()

        # Apply move
        for _ in range(rotations):
            test_state.rot_cw()
        if translation < 0:
            for _ in range(abs(translation)):
                test_state.move_left()
        else:
            for _ in range(translation):
                test_state.move_right()

        # Drop and evaluate
        while not test_state.check_collision():
            test_state.move_down()
        test_state.move_up()
        test_state.place_piece_on_board()

        # Get Q-value
        state_tensor = self._encode_state(test_state)
        with torch.no_grad():
            q_value = self.model(state_tensor).item()

        return q_value
```

### Running Neural Bots

Once implemented, usage matches your existing workflow:

```bash
# Train supervised bot
python bot/neural_bot/training/train_supervised.py --games 1000 --epochs 100

# Run supervised bot
python run.py --bot-model SupervisedBot --model-path bot/neural_bot/checkpoints/supervised_best.pth

# Train DQN bot
python bot/neural_bot/training/train_dqn.py --episodes 10000 --replay-size 20000

# Run DQN bot (no exploration)
python run.py --bot-model DQNBot --model-path bot/neural_bot/checkpoints/dqn_best.pth

# Run DQN bot with exploration (for testing)
python run.py --bot-model DQNBot --model-path bot/neural_bot/checkpoints/dqn_best.pth --epsilon 0.1
```

---

## Technical Considerations

### Computational Requirements

**Phase 1 (Supervised)**:
- Training time: 1-2 hours (CPU)
- Memory: ~2GB
- GPU: Optional, 10x speedup

**Phase 2 (DQN)**:
- Training time: 12-24 hours (CPU), 4-8 hours (GPU)
- Memory: ~4GB (replay buffer)
- GPU: Highly recommended for Phase 2+

**Phase 3 (Advanced)**:
- Training time: 24-72 hours (GPU recommended)
- Memory: 8-16GB (larger replay buffers)
- GPU: Essential for CNN and recurrent models

### PyTorch Best Practices

1. **Device Management**: Always use `.to(device)` for CPU/GPU compatibility
2. **Batch Processing**: Process multiple moves in parallel when possible
3. **Mixed Precision**: Use `torch.cuda.amp` for faster training on modern GPUs
4. **Gradient Clipping**: Prevent exploding gradients in DQN training
5. **Learning Rate Scheduling**: Reduce LR when performance plateaus
6. **Checkpointing**: Save models every N episodes, keep best performer

### Debugging Neural Networks

Common issues and solutions:

**Problem**: Network outputs constant values
- **Cause**: Dead neurons or too-small learning rate
- **Solution**: Check activations, increase LR, use different initialization

**Problem**: Training loss decreases but evaluation performance doesn't improve
- **Cause**: Overfitting to replay buffer
- **Solution**: Increase epsilon, larger replay buffer, more diverse training

**Problem**: Q-values explode
- **Cause**: No gradient clipping or target network updates
- **Solution**: Clip gradients to [-1, 1], update target network more frequently

**Problem**: Agent learns suboptimal policy
- **Cause**: Poor reward shaping
- **Solution**: Experiment with different reward functions, add shaping terms

### Evaluation and Benchmarking

Compare all models on consistent metrics:

**Performance Metrics**:
- Average lines cleared over 100 games
- Average score over 100 games
- Max lines cleared in any game
- Games played before first 100+ line game
- Variance in performance (consistency)

**Comparison Baseline**:
- RandomBot: ~30 lines
- WeightedBotLines (your current best): ~??? lines (measure this first)
- Human expert: 200-300 lines (level 19)

**Success Criteria**:
- Phase 1: 70-90% of WeightedBot performance
- Phase 2: Match or exceed WeightedBot
- Phase 3: Significantly exceed WeightedBot (1.5-2x lines cleared)

---

## Risk Mitigation

### Potential Challenges

1. **Long Training Times**
   - Mitigation: Start with Phase 1, validate infrastructure quickly
   - Use smaller networks initially, scale up gradually
   - Leverage parallelization in your existing code

2. **Hyperparameter Sensitivity**
   - Mitigation: Start with published hyperparameters from literature
   - Use Weights & Biases or TensorBoard for experiment tracking
   - Grid search only after basic version works

3. **Reward Engineering Difficulty**
   - Mitigation: Start simple (lines cleared), iterate based on results
   - Can bootstrap with your GA-discovered features as auxiliary rewards
   - Test reward functions with random bot first (sanity check)

4. **Network May Not Converge**
   - Mitigation: Phase 1 validates pipeline before RL complexity
   - DQN has well-known stability tricks (target network, replay buffer)
   - Can fall back to supervised learning if RL proves too difficult

5. **Performance May Not Exceed GA**
   - Mitigation: Even matching GA proves concept, enables future work
   - Supervised learning provides lower bound
   - Can combine: use NN as evaluation function for GA

---

## Future Extensions

Once basic DQN is working, many exciting directions:

1. **Transfer Learning**: Train on level 19, fine-tune for other levels
2. **Multi-Task Learning**: Single network plays multiple game modes
3. **Meta-Learning**: Network that adapts quickly to new piece distributions
4. **Imitation + RL**: Pre-train with supervised learning, fine-tune with RL
5. **Curriculum Learning**: Start training on easier levels, gradually increase
6. **Self-Play Improvements**: Learn by trying to place difficult boards for itself
7. **Interpretability**: Visualize what CNN filters learn, attention mechanisms
8. **Deployment**: Optimize for real-time performance with ONNX or TorchScript

---

## Conclusion

This plan provides a concrete, research-backed roadmap for building a neural network Tetris AI:

**Phase 1** validates your infrastructure quickly with supervised learning, providing a performance baseline and ensuring all components work before tackling the harder RL problem.

**Phase 2** implements the industry-standard DQN approach, which has proven successful across many Tetris implementations. This is where the real learning happens—the network discovers optimal strategies through millions of games of self-play.

**Phase 3** explores cutting-edge techniques that have pushed DQN to state-of-the-art performance across many domains. Each improvement is modular and builds on the previous work.

The progression from simple to complex allows you to:
- Learn PyTorch and neural network debugging on easier problems
- Validate each component before adding complexity
- Achieve working results quickly (Phase 1) while building toward research-grade performance (Phase 3)
- Make informed decisions about which complexities are worth the effort

Starting with simple feedforward networks and progressing to CNNs, recurrent networks, and Rainbow DQN mirrors how the field itself evolved, giving you intuition for why each technique matters.

The beauty of this approach: even if you never move past Phase 2, you'll have a working neural network that discovers Tetris strategies from scratch—something fundamentally different from your GA approach and a meaningful research contribution.

---

## References and Further Reading

### Key Papers
- Mnih et al. (2015): "Human-level control through deep reinforcement learning" - Original DQN paper
- Van Hasselt et al. (2016): "Deep Reinforcement Learning with Double Q-learning" - Double DQN
- Wang et al. (2016): "Dueling Network Architectures for Deep Reinforcement Learning" - Dueling DQN
- Schaul et al. (2016): "Prioritized Experience Replay" - PER
- Hessel et al. (2018): "Rainbow: Combining Improvements in Deep Reinforcement Learning" - Rainbow DQN

### Tetris-Specific Resources
- [Stanford CS231n: Playing Tetris with Deep RL](https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf)
- [How I trained a Neural Network to Play Tetris](https://timhanewich.medium.com/how-i-trained-a-neural-network-to-play-tetris-using-reinforcement-learning-ecfa529c767a)
- [The Making of a Deep Learning Tetris AI](https://medium.com/wenqins-blog/the-making-of-a-deep-learning-tetris-ai-d5663f21d847)

### Code Repositories
- [vietnh1009/Tetris-deep-Q-learning-pytorch](https://github.com/vietnh1009/Tetris-deep-Q-learning-pytorch)
- [nuno-faria/tetris-ai](https://github.com/nuno-faria/tetris-ai)
- [jaybutera/tetrisRL](https://github.com/jaybutera/tetrisRL)

### PyTorch Resources
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

---

**Document Version**: 1.0
**Last Updated**: 2025-11-25
**Author**: Claude (Anthropic)
**Project**: TetrisAI Neural Network Implementation
