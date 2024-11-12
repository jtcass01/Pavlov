from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed import Stream, DataFeed
from typing import List


class MultiAssetPBR(TensorTradeRewardScheme):
    """A reward scheme for position-based returns for multiple assets.

    * Let :math:`p_t^i` denote the price of asset i at time t.
    * Let :math:`x_t^i` denote the position on asset i at time t.
    * Let :math:`R_t` denote the reward at time t.

    Then the reward is defined as,
    :math:`R_{t} = \sum_i (p_{t}^i - p_{t-1}^i) \cdot x_{t}^i`.

    Parameters
    ----------
    price_streams : List[`Stream`]
        The list of price streams to use for computing rewards.
    """

    registered_name = "multi_asset_pbr"

    def __init__(self, price_streams: List['Stream']) -> None:
        super().__init__()

        # Initialize positions for each asset
        self.price_streams: List[Stream] = price_streams
        self.positions: List[int] = [-1] * len(price_streams)  # -1: short, 1: long
        
        # Create position streams
        self.position_streams: List[Stream] = [
            Stream.sensor(self, lambda rs, i=i: rs.positions[i], dtype="float")
            for i in range(len(self.price_streams))
        ]

        # Create sensors to compute the price differences and positions
        rewards = []
        for i, price in enumerate(self.price_streams):
            price_diff = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
            position_sensor = self.position_streams[i]
            reward = (position_sensor * price_diff).fillna(0).rename(f"reward_{i}")
            rewards.append(reward)

        # Compile the feed with all the rewards
        self.feed = DataFeed(rewards)
        self.feed.compile()


        for i, price in enumerate(self.price_streams):
            if price.value is None:
                print(f"Warning: price value is None for stream {i}")

        for i, position in enumerate(self.position_streams):
            if position.value is None:
                print(f"Warning: position value is None for stream {i}")

    def on_action(self, action: List[int]) -> None:
        """Updates the positions based on the actions.
        
        Parameters
        ----------
        action : List[int]
            A list of actions for each asset. Each action should be 0 (short) or 1 (long).
        """
        self.positions = [-1 if a == 0 else 1 for a in action]

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Calculates the reward as the sum of rewards for all assets."""
        rewards = self.feed.next()
        return sum(rewards[f"reward_{i}"] for i in range(len(self.price_streams)))

    def reset(self) -> None:
        """Resets the positions and the feed."""
        self.positions = [-1] * len(self.price_streams)
        self.feed.reset()
