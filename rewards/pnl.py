from typing import List

from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream
from tensortrade.oms.wallets import Wallet, Portfolio

class PnLRewardScheme(TensorTradeRewardScheme):
    """A reward scheme based on profit and loss."""
    registered_name = "pnl"

    def __init__(self, price: 'Stream', cash_wallet: 'Wallet', asset_wallet: 'Wallet') -> None:
        super().__init__()

        self.price: 'Stream' = price

        self.cash_wallet = cash_wallet
        self.asset_wallet = asset_wallet
        self.previous_value = self._portfolio_value

    @property
    def _portfolio_value(self):
        price: float = self.price.forward()
        return self.cash_wallet.total_balance.as_float() + (self.asset_wallet.total_balance.as_float() * price)

    def on_action(self, action: int) -> None:
        pass
    
    def get_reward(self, portfolio: 'Portfolio') -> float:
        current_value: float = self._portfolio_value
        reward: float = current_value - self.previous_value
        self.previous_value = current_value
        return reward

    def reset(self) -> None:
        self.previous_value = self._portfolio_value

class MutliAssetPnLRewardScheme(TensorTradeRewardScheme):
    """A reward scheme based on profit and loss for multiple assets."""
    registered_name = "multi_asset_pnl"

    def __init__(self, price_streams: List['Stream'], cash_wallet: 'Wallet', asset_wallets: List['Wallet']) -> None:
        super().__init__()

        # Price streams for each asset
        self.price_streams = price_streams
        
        # Cash wallet and multiple asset wallets
        self.cash_wallet = cash_wallet
        self.asset_wallets = asset_wallets
        
        # Store the initial portfolio value
        self.previous_value = self._portfolio_value

    @property
    def _portfolio_value(self) -> float:
        """Calculate the total portfolio value."""
        # Cash balance
        total_value = self.cash_wallet.total_balance.as_float()

        # Add the value of each asset based on its current price
        for asset_wallet, price_stream in zip(self.asset_wallets, self.price_streams):
            asset_balance = asset_wallet.total_balance.as_float()
            current_price = price_stream.forward()
            total_value += asset_balance * current_price
        
        return total_value

    def on_action(self, action: int) -> None:
        """Optional method to react to actions."""
        pass

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Calculate the reward based on the change in portfolio value."""
        current_value = self._portfolio_value
        reward = current_value - self.previous_value
        self.previous_value = current_value
        return reward

    def reset(self) -> None:
        """Reset the previous portfolio value."""
        self.previous_value = self._portfolio_value