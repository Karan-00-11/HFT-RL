import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time
import enum

class Action(enum.Enum):
    GO_SHORT = 0
    HOLD = 1
    GO_LONG = 2
    STOP_LOSS = 3

    def __int__(self):
        return self.value

class TradingEnv(gym.Env):
    """
    A custom Reinforcement Learning environment for financial trading,
    compatible with Gymnasium.

    The agent learns to make discrete trading decisions (Buy, Sell, Hold)
    based on historical OHLCV data and its current portfolio.
    """
    # Metadata for Gymnasium environment
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_cash=10000, transaction_rate=0.001, episode_length=1000):
        """
        Initializes the trading environment.

        Args:
            df (pd.DataFrame): Historical market data (OHLCV).
            initial_cash (float): Starting cash for the agent.
            commission_rate (float): Transaction cost as a percentage of trade value.
        """

        super().__init__()
        self.df = df
        self.initial_cash = initial_cash
        self.transaction_rate = transaction_rate
        init_start = time.time()

        # Internal state variables for the agent's portfolio and simulation progress
        # Environment Parameters
        self.current_step = 0
        self.current_ask_price = 0.0  # Initialize price
        self.current_bid_price = 0.0  # Initialize price
        self.entry_price = None
        self.position = 0.0
        self.inventory = 0.0
        self.cash_in_hand = initial_cash
        self.net_worth = initial_cash
        self.max_net_worth = initial_cash
        self.sequence_length = 10
        self.episode_length = episode_length
        self.max_position = 100 # Maximum number of shares can be held

        #Order Parameters
        self.size_pct_of_level = 0.3 # Planning to use Kelly Criterion for order size
        self.current_bids = 0
        self.current_asks = 0

        # Frame Setup
        self.no_of_features = 7 # [1] Bid price, [2] Bid volume, [3] Ask price, [4] Ask volume, [5] Mid price, [6] Spread, [7] Micro price
        self.dom_shape =  (self.sequence_length, self.no_of_features) # Depth of market (DOM) features        
        self.single_frame_size = np.prod(self.dom_shape)  # Total size of a single frame

        # Define action space: Discrete actions for simplicity [1]
        # 0: Hold, 1: Buy (using all available cash), 2: Sell (all held shares)
        self.action_space = spaces.Discrete(4)

        spaces = {
            'cash_in_hand' : spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'position': spaces.Box(low=-1, high=1, shape=(1, 1)),
            'inventory' : spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'Bid price' : spaces.Box(low=0, high=np.inf, shape=(self.sequence_length,), dtype=np.float32),
            'Bid volume' : spaces.Box(low=0, high=np.inf, shape=(self.sequence_length,), dtype=np.float32),
            'Ask price' : spaces.Box(low=0, high=np.inf, shape=(self.sequence_length,), dtype=np.float32),
            'Ask volume' : spaces.Box(low=0, high=np.inf, shape=(self.sequence_length,), dtype=np.float32),
            'Mid price' : spaces.Box(low=0, high=np.inf, shape=(self.sequence_length,), dtype=np.float32),
            'Spread' : spaces.Box(low=0, high=np.inf, shape=(self.sequence_length,), dtype=np.float32),
            'Micro price' : spaces.Box(low=0, high=np.inf, shape=(self.sequence_length,), dtype=np.float32),
            'Order Imbalance' : spaces.Box(low=-1, high=1, shape=(self.sequence_length,), dtype=np.float32)
        }
        ## Observation Space - (sequence_length, features)
        self.observation_space = gym.spaces.Dict(spaces)

        self.frame = None 
            
    def _get_mid_price(self, bid_p, ask_p):
        mid_price = (bid_p + ask_p)/2.0
        return mid_price


    def _get_micro_price(self, bid_p, bid_v, ask_p, ask_v):
        total_vol = bid_v + ask_v
        micro_price = np.where(total_vol > 0.0,
                               (bid_p * ask_v + ask_p * bid_v)/total_vol,
                               (bid_p + ask_p) / 2.0)
        
        return micro_price
    
    def _order_imbalance(self, bid_v, ask_v):
        total_volume = bid_v + ask_v
        # Avoid division by zero
        imbalance = np.where(total_volume > 0.0, 
                            (bid_v - ask_v) / total_volume,
                            0)
        return imbalance
    
    def _spread(self, bid_p, ask_p):
        spread = ask_p - bid_p
        return spread
    
    def features(self):

        self.frame = self.df.iloc[self.current_step : self.current_step + self.sequence_length, :-1]                
        return self.frame
    
    def _get_obs(self):
        """
        Constructs the observation array for the current step.
        This is the 'state' the agent observes.[2]
        """
        features_df = self.features()
        features_df = features_df.values.astype(np.float32)
        return {
        'cash_in_hand': np.array([self.cash_in_hand], dtype=np.float32),
        'position': np.array([self.position], dtype=np.float32),
        'inventory': np.array([self.inventory], dtype=np.float32),
        'Bid price': features_df[:, 0],
        'Bid volume': features_df[:, 2], 
        'Ask price': features_df[:, 1],
        'Ask volume': features_df[:, 3],
        'Mid price': self._get_mid_price(features_df[:, 0], features_df[:, 1]), 
        'Spread': self._spread(features_df[:, 0], features_df[:, 1]),
        'Micro price': self._get_micro_price(features_df[:, 0], features_df[:, 2], features_df[:, 1], features_df[:, 3]),
        'Order Imbalance': self._order_imbalance(features_df[:, 2], features_df[:, 3])
    }
    
    def _update_market_state(self):
        """
        Fetches the current market data from the dataframe for the current step.
        """
        # Get the entire row of data for the current time step
        current_data_row = self.df.iloc[self.current_step]
        
        # Set the bid and ask prices from that row's data
        self.current_bid_price = current_data_row['Bid price']
        self.current_ask_price = current_data_row['Ask price']

        self.current_price = (self.current_bid_price + self.current_ask_price) / 2

    def _execute_action(self, action, quantity=1):
        reward = 0.0
        closed = False

        # Action 0: open short OR close long
        if action == Action.GO_SHORT.value:
            if self.position == 0:  # neutral -> open short
                self.position = -1
                self.inventory -= quantity
                self.cash_in_hand += quantity * self.current_bid_price
                self.entry_price = self.current_bid_price
            elif self.position == 1:  # long -> close long
                reward = (self.current_bid_price - self.entry_price) * self.inventory
                self.cash_in_hand += self.inventory * self.current_bid_price
                self.inventory = 0
                self.position = 0
                self.entry_price = None
                closed = True
            else:  # already short
                reward -= 0.001  # penalty

        # Action 2: open long OR close short
        elif action == Action.GO_LONG.value:
            if self.position == 0:  # neutral -> open long
                cost = quantity * self.current_ask_price
                if self.cash_in_hand >= cost:
                    self.position = 1
                    self.inventory += quantity
                    self.cash_in_hand -= cost
                    self.entry_price = self.current_ask_price
            elif self.position == -1:  # short -> close short
                reward = (self.entry_price - self.current_ask_price) * abs(self.inventory)
                cost_to_close = abs(self.inventory) * self.current_ask_price
                self.cash_in_hand -= cost_to_close
                self.inventory = 0
                self.position = 0
                self.entry_price = None
                closed = True
            else:  # already long
                reward -= 0.001

        # Action 3: stop-loss (close if losing)
        elif action == Action.STOP_LOSS.value and self.position != 0 and self.entry_price is not None:
            unrealized_pnl = (self.current_price - self.entry_price) * self.inventory
            # Inside the if unrealized_pnl < 0: block
            if self.position > 0: # Closing a long
                self.cash_in_hand += self.inventory * self.current_bid_price
            else: # Closing a short
                cost_to_close = abs(self.inventory) * self.current_ask_price
                self.cash_in_hand -= cost_to_close

            self.inventory = 0 # Reset inventory after updating cash
            if unrealized_pnl < 0:
                # The reward is the realized loss (including transaction costs for closing)
                transaction_cost = (self.entry_price + (self.current_ask_price if self.position < 0 else self.current_bid_price)) * self.transaction_rate
                reward = unrealized_pnl - transaction_cost
                self.cash_in_hand += self.inventory * self.current_price
                self.position = 0
                closed = True

        # Action 1 = hold → do nothing

        return reward, closed

    def _get_info(self):
        """
        Provides additional, environment-specific information useful for debugging
        or understanding the environment's internal state.[4]
        """
        return {
            'step' : self.current_step,
            'cash_in_hand': self.cash_in_hand,
            'shares_held': self.position,
            'net_worth': self.net_worth,
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new episode.[4]
        """
        super().reset(seed=seed) # Important for Gymnasium's seeding

        # Reset all internal state variables
        self.current_step = 0
        self._update_market_state()
        self.position = 0.0
        self.entry_price = None
        self.cash_in_hand = self.initial_cash
        self.returns = [self.cash_in_hand]  # Track returns over time
        self.net_worth = self.initial_cash
        self.current_price = 0.0 
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # 1. Update current market price
        self._update_market_state()

        # 2. Save previous net worth
        prev_worth = self.net_worth

        # 3. Execute action (trading logic)
        action_reward, closed = self._execute_action(action)

        # 4. Mark-to-market unrealized PnL
        # unrealized_pnl = (self.current_price - self.entry_price) * self.inventory if (
        #     self.inventory != 0 and self.entry_price is not None
        # ) else 0.0

        # 5. Update net worth
        self.net_worth = self.cash_in_hand + self.inventory * self.current_price

        # 6. Reward: realized action reward + scaled net worth change
        pnl_reward = (self.net_worth - prev_worth) / prev_worth if prev_worth > 0 else 0.0
        # reward = (self.net_worth - prev_worth) / prev_worth
        reward = action_reward + pnl_reward * 100  # scaling factor

        # 7. Advance time
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        # 8. Build observation
        obs = self._get_obs()

        # 9. Extra info for debugging
        info = {
            "cash": self.cash_in_hand,
            "inventory": self.inventory,
            "position_dir": self.position,
            "entry_price": self.entry_price,
            "net_worth": self.net_worth,
            "closed": closed,
            "action": action,
        }

        return obs, reward, terminated, truncated, info


    def render(self, mode='human'):
        """
        Renders the environment state. For a trading environment, this typically
        involves printing key metrics or plotting visualizations.
        """
        # Simple print for demonstration. For advanced visualization, use libraries like Matplotlib.
        print(f"Step: {self.current_step}, Net Worth: ${self.net_worth:.2f}, Cash: ${self.cash_in_hand:.2f}, Shares: {self.position}")

    def close(self):
        """
        Cleans up any resources used by the environment.
        """
        pass # No specific resources to close for this simple example
