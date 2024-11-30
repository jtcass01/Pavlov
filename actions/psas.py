from typing import List
from tensortrade.env.default.actions import TensorTradeActionScheme
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.orders import Order, proportion_order
from gym.spaces import Box
from numpy import float32, ndarray

class PSAS(TensorTradeActionScheme):
    """A position scaling action scheme where actions correspond to scaling buy/sell positions.
    
    This scheme uses a continuous action space to adjust positions dynamically.
    
    Parameters
    ----------
    cash : `Wallet`
        The wallet holding funds in the base instrument (e.g., USD).
    asset : `Wallet`
        The wallet holding the asset being traded (e.g., BTC).
    max_scale : float
        The maximum proportion of the wallet's balance to use for buying/selling.
    """

    registered_name = "psas"

    def __init__(self, cash: 'Wallet', asset: 'Wallet', max_scale: float = 0.5):
        super().__init__()
        self.cash = cash  # Wallet holding cash for buying assets
        self.asset = asset  # Wallet holding the tradable asset
        self.max_scale = max_scale  # Maximum proportion of wallet to trade
        self.current_position = 0.0  # Current asset position (0 = no asset)
    
    @property
    def action_space(self):
        """Defines a continuous action space ranging from -1 (full sell) to 1 (full buy)."""
        return Box(low=-1, high=1, shape=(1,), dtype=float)

    def get_orders(self, action: float, portfolio: 'Portfolio') -> 'Order':
        """Generate orders based on the continuous action value.
        
        Parameters
        ----------
        action : float
            A continuous value between -1 and 1 indicating the proportion of the asset to buy or sell.
        portfolio : `Portfolio`
            The portfolio containing all the wallets.
        
        Returns
        -------
        List of `Order`
            The generated buy/sell orders based on the action.
        """
        orders = []
        # Calculate the proportion to trade based on the action and max_scale
        scale = action[0] * self.max_scale
        
        if scale > 0:  # Buying assets
            amount_to_buy = scale * self.cash.balance
            if amount_to_buy > 0:
                orders.append(proportion_order(portfolio, self.cash, self.asset, scale))
        
        elif scale < 0:  # Selling assets
            amount_to_sell = abs(scale) * self.asset.balance
            if amount_to_sell > 0:
                orders.append(proportion_order(portfolio, self.asset, self.cash, abs(scale)))

        # Update the current position based on the action taken
        self.current_position = 1 if scale > 0 else -1 if scale < 0 else 0
        return orders

    def reset(self):
        """Resets the internal state of the action scheme."""
        super().reset()
        self.current_position = 0.0