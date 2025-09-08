# HFT-RL: High-Frequency Trading with Reinforcement Learning

A reinforcement learning framework for high-frequency trading strategies using limit order book data and custom Gymnasium environments.

## Features

- **Custom Trading Environment**: OpenAI Gymnasium-compatible environment for HFT simulation
- **Multi-feature State Representation**: Includes bid/ask prices, volumes, spreads, and micro-prices
- **Sliding Window Observations**: Temporal sequence modeling for better market prediction
- **Realistic Trading Mechanics**: Transaction costs, position limits, and portfolio management
- **Multiple RL Algorithms**: Support for PPO, A2C, and other Stable Baselines3 algorithms
- **Data Pipeline**: Efficient processing of large-scale LOB datasets using Pandas/Polars

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Karan-00-11/HFT-RL.git
cd HFT-RL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from envs.env import TradingEnv
from stable_baselines3 import PPO

# Load your market data
df = pd.read_csv("data/USATECHIDXUSD_mt5_ticks_2.csv")

# Create trading environment
env = TradingEnv(df, episode_length=1000)

# Train RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate trained agent
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Training with Vectorized Environments

```python
from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = DummyVecEnv([lambda: TradingEnv(df, episode_length=100000)])
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
```

## Project Structure

```
HFT-RL/
├── src/
│   ├── train.ipynb          # Training and evaluation notebook
│   ├── agents/              # RL agent configurations
│   └── envs/
│       ├── env.py           # Main trading environment
│       └── __pycache__/
├── data/
│   └── USATECHIDXUSD_mt5_ticks_2.csv  # Sample LOB data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Dependencies

- gymnasium
- stable-baselines3
- sb3-contrib
- pandas
- polars
- numpy
- matplotlib

## Environment Details

### Observation Space
- **Shape**: `(sequence_length * no_of_features,)` - flattened temporal features
- **Features per timestep**:
  - Bid price and volume
  - Ask price and volume
  - Mid price
  - Bid-ask spread
  - Micro price

### Action Space
- **Type**: Discrete
- **Actions**: Buy, Sell, Hold with variable position sizes

### Reward Function
- Based on portfolio value changes
- Includes transaction cost penalties
- Penalizes excessive position taking

## Data Format

The environment expects CSV data with the following columns:
- `Timestamp`: DateTime string
- `Bid price`: Best bid price
- `Bid volume`: Best bid volume
- `Ask price`: Best ask price
- `Ask volume`: Best ask volume

## Configuration

Key parameters in `TradingEnv`:
- `sequence_length`: Number of timesteps in observation window
- `initial_cash`: Starting portfolio value
- `transaction_rate`: Trading fee percentage
- `episode_length`: Maximum steps per episode

