import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time

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
        self.current_price = 0.0  # Initialize price
        self.position = 0.0
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
        self.action_space = spaces.Discrete(3)

        ## Observation Space - (sequence_length, features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.sequence_length * self.no_of_features,), dtype=np.float32)


        # self.frame_buffer = deque(maxlen=self.sequence_length)  # Buffer to hold the last 'sequence_length' frames
        self.frame = np.zeros(self.dom_shape, dtype=np.float32)  # Initialize a frame with zeros


    def _update_market_data(self):
        """Update current bids and asks from the dataframe at the current step"""
        if self.current_step >= len(self.df):
            return  # Prevent index errors
            
        row = self.df.iloc[self.current_step]
        
        # Extract bid/ask data for the current step
        bids = [[row['Bid price'], row['Bid volume']]]
        asks = [[row['Ask price'], row['Ask volume']]]
        
        self.current_bids = np.array(bids, dtype=np.float32)
        self.current_asks = np.array(asks, dtype=np.float32)
        
        # Calculate current price (mid price for valuation)
        if len(self.current_bids) > 0 and len(self.current_asks) > 0:
            self.current_price = self.get_mid_price(self.current_bids, self.current_asks)
        else:
            # If no market data, use the last known price or a default
            self.current_price = getattr(self, 'current_price')
            
    def get_mid_price(self, bids, asks):
        best_bid = bids[0][0] if len(bids) > 0 else 0.0
        best_ask = asks[0][0] if len(asks) > 0 else 0.0
        return (best_bid + best_ask) / 2.0


    def get_micro_price(self, bids, asks):
        if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
            return self.get_mid_price(bids, asks)
        bid_price, bid_vol = bids[0]
        ask_price, ask_vol = asks[0]
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (bid_price + ask_price) / 2.0
        return (bid_price * ask_vol + ask_price * bid_vol) / total_vol

    def _get_obs(self):
        """
        Constructs the observation array for the current step.
        This is the 'state' the agent observes.[2]
        """
        ### I just dont know what Im doing here
        # stacked_frames = self.frame


        # if stacked_frames.shape[0] < self.sequence_length:
        #     last_frame = stacked_frames[-1] if len(stacked_frames) > 0 else np.zeros(self.single_frame_size)
        #     padding_needed = self.sequence_length - stacked_frames.shape[0]
        #     padding = np.tile(last_frame[np.newaxis, :], (padding_needed, 1))
        #     stacked_frames = np.vstack((stacked_frames, padding))
        
        # return stacked_frames
        return self.features().flatten().astype(np.float32)
    
    def features(self):
        bids = self.df.loc[self.current_step : self.current_step + self.sequence_length + 1, 
                        ['Bid price', 'Bid volume']].to_numpy()
        asks = self.df.loc[self.current_step : self.current_step + self.sequence_length + 1, 
                        ['Ask price', 'Ask volume']].to_numpy()
        # remaining_steps = (len(self.df) - self.sequence_length) - self.current_step 
        
        # if remaining_steps >= self.sequence_length:
        #     # We have enough data
        #     bids = self.df.loc[self.current_step:self.current_step + self.sequence_length - 1, 
        #                     ['Bid price', 'Bid volume']].to_numpy()
        #     asks = self.df.loc[self.current_step:self.current_step + self.sequence_length - 1, 
        #                     ['Ask price', 'Ask volume']].to_numpy()
        # else:
        #     # Not enough data, need padding
        #     available_rows = self.df.loc[self.current_step:, 
        #                                 ['Bid price', 'Bid volume', 'Ask price', 'Ask volume']]
            
        #     # Get last row for padding
        #     last_row = available_rows.iloc[-1]
        #     last_bids = np.array([[last_row['Bid price'], last_row['Bid volume']]])
        #     last_asks = np.array([[last_row['Ask price'], last_row['Ask volume']]])
            
        #     # Create arrays with available data
        #     bids = available_rows[['Bid price', 'Bid volume']].to_numpy()
        #     asks = available_rows[['Ask price', 'Ask volume']].to_numpy()
            
        #     # Calculate padding needed
        #     padding_needed = self.sequence_length - remaining_steps
            
        #     # Pad with copies of the last row
        #     pad_bids = np.tile(last_bids, (padding_needed, 1))
        #     pad_asks = np.tile(last_asks, (padding_needed, 1))
            
        #     # Concatenate available data with padding
        #     bids = np.vstack((bids, pad_bids))
        #     asks = np.vstack((asks, pad_asks))
        
        for i in range(self.sequence_length):
            self.frame[i, 0] = bids[i][0]
            self.frame[i, 1] = bids[i][1]
            self.frame[i, 2] = asks[i][0]
            self.frame[i, 3] = asks[i][1]
            self.frame[i, 4] = (bids[i][0] + asks[i][0])/2
            self.frame[i, 5] = asks[i][0] - bids[i][0]     
            bid_vol, ask_vol = bids[i][1], asks[i][1]
            total_vol = bid_vol + ask_vol
            if total_vol == 0:
                self.frame[i, 6] = self.frame[i, 4]  # Use mid price if volumes are 0
            else:
                self.frame[i, 6] = (bids[i][0] * ask_vol + asks[i][0] * bid_vol) / total_vol
                
        return self.frame.flatten()

    def _initial_frame(self):
        """
        Initializes the frame buffer with the first observation.
        This is called at the start of each episode to set the initial state.[2]
        """

        return self.features().astype(np.float32)
  
    def _get_current_frame(self):
        """
        Constructs the current frame observation for the agent.
        This includes the current cash, shares held, and market data.[2]
        """
        if self.current_step == 0:
            self._initial_frame()
        
        return self.features()

    
    def _market_buy(self):

        if len(self.current_bids) == 0:
            return -0.1
        
        ask_price = self.current_asks[0][0]  # Get the best ask price
        ask_volume = self.current_asks[0][1]  # Get the best ask volume
        
        ### Need to set Kelly's Criterion dynamically, but for now we set it as 0.1
        order_size = self.size_pct_of_level * ask_volume  # Calculate order size based on the percentage of the best ask volume
        if self.position + order_size > self.max_position:
            return -0.05

        if self.cash_in_hand < order_size * ask_price or self.cash_in_hand <= 0:
            return -np.inf
        
        trade_value = order_size * ask_price
        tranaction_cost = trade_value * self.transaction_rate

        self.position += order_size
        self.cash_in_hand -= (trade_value + tranaction_cost)

        return -tranaction_cost  # Return the negative transaction cost as a penalty
    
    def _market_sell(self):
        if len(self.current_bids) == 0:
            return -0.1
        
        bid_price = self.current_bids[0][0]  # Get the best bid price
        bid_volume = self.current_bids[0][1]  # Get the best bid volume
        order_size = self.size_pct_of_level * bid_volume  # Calculate order size based on the percentage of the best bid volume
        if self.position - order_size < 0:
            return -0.05
        trade_value = order_size * bid_price
        transaction_cost = trade_value * self.transaction_rate
        self.position -= order_size
        self.cash_in_hand += (trade_value - transaction_cost)
        return -transaction_cost  # Return the negative transaction cost as a penalty


    def _execute_action(self, action):
        """
        Executes the action chosen by the agent.
        This updates the agent's portfolio and market state based on the action taken.[2]
        
        Args:
            action (int): The action chosen by the agent (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
            float: The reward for the action taken.
        """
        # if action == 0:
        #     # Hold action, no changes to portfolio
        #     reward = -1
        #     return reward
        # elif action == 1:
        #     # Buy action
        #     reward = self._market_buy()
        #     return reward
        # elif action == 2:
        #     # Sell action
        #     reward = self._market_sell()
        #     return reward
    # Store previous net worth to calculate PnL
        prev_worth = self.net_worth
        
        # Execute the action
        if action == 0:
            # Smaller penalty for holding
            action_reward = -0.01
        elif action == 1:
            action_reward = self._market_buy()
        elif action == 2:
            action_reward = self._market_sell()
        
        # Calculate new net worth after action
        self.net_worth = self.cash_in_hand + (self.position * self.current_price)
        
        # Calculate PnL component - the real trading reward
        pnl_reward = (self.net_worth - prev_worth) / prev_worth if prev_worth > 0 else 0
        
        # Scale PnL reward to make it more significant
        scaled_pnl = pnl_reward * 100
        
        # Total reward combines action cost and PnL
        return scaled_pnl + action_reward

    def _get_info(self):
        """
        Provides additional, environment-specific information useful for debugging
        or understanding the environment's internal state.[4]
        """
        return {
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
        self.position = 0.0
        self.cash_in_hand = self.initial_cash
        self.returns = [self.cash_in_hand]  # Track returns over time
        self.net_worth = self.initial_cash
        self.current_price = 0.0 
        self._update_market_data()
        # self.frame_buffer.clear()  # Clear the frame buffer for new episode
        self.frame = self._initial_frame()
        # Get the initial observation and info
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Takes an action chosen by the agent and advances the environment by one step.[4]

        Args:
            action (int): The action chosen by the agent (0: Hold, 1: Buy, 2: Sell).

        Returns:
            tuple: (observation, reward, done, truncated, info)
                observation (np.array): The new state of the environment.
                reward (float): The reward received for the action.
                done (bool): True if the episode has terminated.
                truncated (bool): True if the episode was truncated (e.g., time limit reached).
                info (dict): Additional debugging information.
        """ 

        # self._update_market_data()
        self._update_market_data()  # Update market data for the current step
        reward = self._execute_action(action)
        current_price_tr = [self.current_price] 
        self.net_worth = self.cash_in_hand + (self.position * self.current_price)  # Update net worth based on current cash and position value
        self.max_net_worth = max(self.net_worth, self.max_net_worth)  # Track maximum net worth
        self.current_step += 1

        terminated = self.current_step >= self.episode_length  # Check if the episode is done
        Truncated = False  # In this simple example, we don't have truncation logic
        # Get the new observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, Truncated, info

    def render(self, mode='human'):
        """
        Renders the environment state. For a trading environment, this typically
        involves printing key metrics or plotting visualizations.
        """
        # Simple print for demonstration. For advanced visualization, use libraries like Matplotlib.
        print(f"Step: {self.current_step}, Net Worth: ${self.net_worth:.2f}, Cash: ${self.cash_in_hand:.2f}, Shares: {self.shares_held}, Trades: {self.trades}")

    def close(self):
        """
        Cleans up any resources used by the environment.
        """
        pass # No specific resources to close for this simple example
