from typing import List

from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.wallets import Portfolio

from numpy import isnan, isfinite

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
        self.position = 0

        # Calculate the portfolio returns
        price_diff: Stream = price.diff().fillna(0)
        volatility: Stream = self.get_volatility(price)
        safe_volatility = volatility.apply(
            lambda v: max(v, 1e-5)
        )
        reward = (price_diff / safe_volatility).rename("reward")
        reward.apply(lambda r: r if not isnan(r) and isfinite(r) else 0)  # Replace NaN/inf with 0

        self.feed = DataFeed([reward])
        self.feed.compile()

    def get_volatility(self, price: 'Stream') -> 'Stream':
        """Calculate rolling standard deviation (volatility) manually."""
        return price.rolling(self.window_size).std() 

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Return the reward."""
        return self.feed.next()["reward"]

    def on_action(self, action: int) -> None:
        """Updates the portfolio position based on the action."""
        self.position = action

    def reset(self) -> None:
        """Reset the reward scheme."""
        self.feed.reset()