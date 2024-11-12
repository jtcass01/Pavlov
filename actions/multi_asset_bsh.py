from gym.spaces import MultiDiscrete
from typing import List, Optional
from tensortrade.env.default.actions import TensorTradeActionScheme
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.orders import Order, proportion_order


class MultiAssetBSH(TensorTradeActionScheme):
    """
    A discrete action scheme that allows buying, selling, or holding multiple assets simultaneously.

    Parameters
    ----------
    cash_wallet : `Wallet`
        The wallet to hold funds in the base currency (e.g., USD).
    asset_wallets : `List[Wallet]`
        A list of wallets to hold funds in various assets (e.g., BTC, ETH, ADA).
    """

    registered_name = "multi_asset_bsh"

    def __init__(self, cash_wallet: Wallet, asset_wallets: List[Wallet]):
        super().__init__()
        self.cash_wallet = cash_wallet
        self.asset_wallets = asset_wallets
        self.n_assets = len(asset_wallets)
        self.action = [0] * self.n_assets  # Initial action is "Hold" for each asset

        self.listeners = []

    @property
    def action_space(self):
        # MultiDiscrete action space where each asset has 3 possible actions: Hold, Buy, Sell
        return MultiDiscrete([3] * self.n_assets)

    def attach(self, listener):
        self.listeners.append(listener)
        return self

    def get_orders(self, actions: List[int], portfolio) -> List[Optional[Order]]:
        """
        Generate orders based on the actions for each asset.
        """
        orders = []

        for i, action in enumerate(actions):
            asset_wallet: Wallet = self.asset_wallets[i]
            current_action: int = self.action[i]

            if action != current_action:  # Action has changed, so perform it
                if action == 1:  # Buy action
                    # Check if there is cash available to buy the asset
                    if self.cash_wallet.balance > 0:
                        order: Order = proportion_order(portfolio, self.cash_wallet, asset_wallet, 1.0)
                        orders.append(order)
                elif action == 2:  # Sell action
                    # Check if there is an asset balance to sell
                    if asset_wallet.balance > 0:
                        order: Order = proportion_order(portfolio, asset_wallet, self.cash_wallet, 1.0)
                        orders.append(order)

                # Update the current action for this asset
                self.action[i] = action

        # Notify listeners of the actions taken
        for listener in self.listeners:
            listener.on_action(actions)

        return orders

    def reset(self):
        super().reset()
        self.action = [0] * self.n_assets  # Reset actions to "Hold" for each asset
