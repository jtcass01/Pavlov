from typing import List

from pandas import DataFrame, Series
from numpy import arange, array
import matplotlib.pyplot as plt
from tensortrade.env.generic import Renderer

from gym import Env

class PositionChangeChart(Renderer):
    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env: Env, **kwargs):
        print("\n\n\n\n RENDERING \n\n\n\n")
        print(kwargs)
        cash_name: str = kwargs['cash_name']
        asset_name: str = kwargs['asset_name']
        
        history: DataFrame = DataFrame(env.observer.renderer_history)
        print(history)
        actions: List[int] = list(history.action)
        p: List[float] = list(history[f"bitfinex:/{cash_name}-{asset_name}"])

        buy = {}
        sell = {}

        for i in range(len(actions) - 1):
            a1 = actions[i]
            a2 = actions[i + 1]

            if a1 != a2:
                if a1 == 0 and a2 == 1:
                    buy[i] = p[i]
                else:
                    sell[i] = p[i]

        buy = Series(buy)
        sell = Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle(f"{kwargs['env_name']} Performance")

        axs[0].plot(arange(len(p)), p, label="price", color=self.color)
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")

        performance_df = DataFrame().from_dict(env.action_scheme.portfolio.performance, orient='index')
        performance_df.plot(ax=axs[1])
        axs[1].set_title("Net Worth")
        
        plt.savefig(f'{kwargs["env_name"]}_trading_performance.png')
        plt.close()


class MultiAssetPositionChangeChart(Renderer):
    def __init__(self, color: str = "orange"):
        self.color = color

    def render(self, env: Env, **kwargs):
        # Extract trading history from the environment
        history = DataFrame(env.observer.renderer_history)

        actions = history.action
        prices: List[DataFrame] = [
            history.eth_price, history.btc_price, history.ada_price,
            history.sol_price, history.ltc_price, history.tron_price
        ]
        
        asset_names: List[str] = [
            "Ethereum", "Bitcoin", "Cardano",
            "Solana", "Litecoin", "Tron"
        ]

        # Retrieve asset wallets and confirm the number of assets matches
        asset_wallets = env.action_scheme.asset_wallets
        n_assets = len(asset_wallets)

        if n_assets != len(prices):
            raise ValueError(f"Mismatch between number of assets ({n_assets}) and price streams ({len(prices)})")

        # Initialize dictionaries for buy and sell signals for each asset
        buys = {i: {} for i in range(n_assets)}
        sells = {i: {} for i in range(n_assets)}

        # Detect buy and sell signals based on action changes
        for i in range(len(actions) - 1):
            for asset_idx in range(n_assets):
                action_current = actions[i][asset_idx]
                action_next = actions[i + 1][asset_idx]

                # Ensure we have valid prices
                if i < len(prices[asset_idx]):
                    price = prices[asset_idx][i]
                else:
                    continue  # Skip if price data is missing

                # Detect buy (0 -> 1) and sell (1 -> 2) transitions
                if action_current != action_next:
                    if action_current == 0 and action_next == 1:  # Buy signal
                        buys[asset_idx][i] = price
                    elif action_current == 1 and action_next == 2:  # Sell signal
                        sells[asset_idx][i] = price

        # Convert buy/sell dictionaries to Series
        buy_series = {i: Series(buys[i]) for i in range(n_assets)}
        sell_series = {i: Series(sells[i]) for i in range(n_assets)}

        # Set up subplots for each asset, plus one for the net worth chart
        num_plots = n_assets + 1  # Include extra plot for portfolio net worth
        cols = 2
        rows = (num_plots + 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
        axs = axs.flatten()
        fig.suptitle(f"{kwargs['env_name']} Multi-Asset Trading Performance", fontsize=16)

        # Plot trading charts for each asset
        for asset_idx in range(n_assets):
            p = prices[asset_idx]
            ax = axs[asset_idx]
            asset_name: str = asset_names[asset_idx]

            # Plot price and buy/sell signals
            ax.plot(arange(len(p)), p, label=f"{asset_name} Price", color=self.color)
            ax.scatter(buy_series[asset_idx].index, buy_series[asset_idx].values, marker="^", color="green", label="Buy")
            ax.scatter(sell_series[asset_idx].index, sell_series[asset_idx].values, marker="v", color="red", label="Sell")
            ax.set_title(f"{asset_name} Trading Chart")
            ax.legend()

        # Plot net worth of the entire portfolio in the last subplot
        performance_df = DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
        print(f'performance_df: {performance_df}')
        performance_ax = axs[-1]
        performance_df.plot(ax=performance_ax, legend=False)
        performance_ax.set_title("Portfolio Net Worth")
        performance_ax.set_xlabel("Time")
        performance_ax.set_ylabel("Net Worth")

        # Hide any unused subplots if the number of assets is odd
        for j in range(n_assets, len(axs) - 1):
            fig.delaxes(axs[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
        plt.savefig(f'{kwargs["env_name"]}_trading_performance.png')
        plt.close()