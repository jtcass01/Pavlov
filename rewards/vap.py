from typing import List

from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.wallets import Portfolio

from numpy import std

class VAP(TensorTradeRewardScheme):
    """A reward scheme that adjusts rewards based on portfolio value changes and market volatility.
    
    Parameters
    ----------
    price : `Stream`
        The price stream of the asset being traded.
    window_size : int
        The size of the rolling window used to calculate volatility.
    """

    registered_name = "vap"

    def __init__(self, price: 'Stream', window_size: int = 30):
        super().__init__()
        self.window_size = window_size
        self.price = price
        self.past_values = []  # To store portfolio values for rolling calculations
        self.position = 0

        # Calculate the portfolio returns
        price_diff = price.diff().fillna(0)
        reward = price_diff / self.get_volatility().rename("reward")
        self.feed = DataFeed([reward])
        self.feed.compile()

    def get_volatility(self) -> Stream:
        """Calculate rolling standard deviation (volatility) manually."""
        def rolling_std(values: List[float], window_size: int) -> float:
            if len(values) < window_size:
                return 0.0
            return std(values[-window_size:])

        return Stream.sensor(self.price, lambda p: rolling_std(p.values, self.window_size), dtype="float").rename("volatility")

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Return the reward."""
        return self.feed.next()["reward"]

    def on_action(self, action: int) -> None:
        """Updates the portfolio position based on the action."""
        self.position = action

    def reset(self) -> None:
        """Reset the reward scheme."""
        self.past_values.clear()
        self.feed.reset()