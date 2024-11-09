from os.path import join
from typing import List
from math import log10

from matplotlib.pyplot import plot, show, title, xlabel, ylabel, legend
from pandas import DataFrame, read_csv, concat
import pandas_ta as ta
from numpy import array, prod, digitize, cumprod, quantile, linspace, zeros, argmax, mean, ndarray
from numpy.random import randint, random

from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.env.default.actions import ManagedRiskOrders, SimpleOrders
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import Instrument


# class DiscretObs():
#     def __init__(self, bins_list: List[array]):
#         self._bins_list = bins_list
#         self._bins_num = len(bins_list)
#         self._state_num_list = [len(bins)+1 for bins in bins_list]
#         self._state_num_total = prod(self._state_num_list)
    
#     @property
#     def state_num_total(self):
#         return self._state_num_total
    
#     @property
#     def state_num_list(self):
#         return self._state_num_list
    
#     def obs2state(self, obs):
#         if not len(obs)==self._bins_num:
#             raise ValueError("observation must have length {}".format(self._bins_num))
#         else:
#             return [digitize(obs[i], bins=self._bins_list[i]) for i in range(self._bins_num)]
        
#     def obs2idx(self, obs):
#         state = self.obs2state(obs)
#         return self.state2idx(state)
    
#     def state2idx(self, state):
#         idx = 0
#         for i in range(self._bins_num-1,-1,-1):
#             idx = idx*self._state_num_list[i]+state[i]
#         return idx
    
#     def idx2state(self, idx):
#         state = [None]*self._bins_num
#         state_num_cumul = cumprod(self._state_num_list)
#         for i in range(self._bins_num-1,0,-1):
#             state[i] = idx//state_num_cumul[i-1]
#             idx -=state[i]*state_num_cumul[i-1]
#         state[0] = idx%state_num_cumul[0]
#         return state
    
# def create_quantile_bins(data, num_bins):
#     """Create quantile-based bins."""
#     return quantile(data.dropna(), q=linspace(0, 1, num_bins + 1)[1:-1])


DATA_DIRECTORY: str = "/home/durzo/Pavlov/data/bitfinex"

bitfinex_btc: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_btc.csv'))
bitfinex_eth: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_eth.csv'))
bitfinex_ada: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_ada.csv'))
bitfinex_sol: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_sol.csv'))
bitfinex_ltc: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_ltc.csv'))
bitfinex_tron: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_tron.csv'))

dfs: List[DataFrame] = [bitfinex_btc, bitfinex_eth, bitfinex_ada,
                        bitfinex_sol, bitfinex_ltc, bitfinex_tron]

bins_list: List[array] = []
for df in dfs:
    df.ta.log_return(append=True, length=16)
    df.ta.rsi(append=True, length=14)
    df.ta.macd(append=True, fast=12, slow=26)
    df.ta.roc(append=True, length=10)
    df.ta.bbands(append=True, length=20, std=2)
    df.ta.obv(append=True)
    df.ta.stoch(append=True, k=14, d=3, smooth_k=3)
    df.dropna(inplace=True)

bitfinex = Exchange("bitfinex", service=execute_order)(
    Stream.source(list(bitfinex_btc['close']), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitfinex_eth['close']), dtype="float").rename("USD-ETH"),
    Stream.source(list(bitfinex_ada['close']), dtype="float").rename("USD-ADA"),
    Stream.source(list(bitfinex_sol['close']), dtype="float").rename("USD-SOL"),
    Stream.source(list(bitfinex_ltc['close']), dtype="float").rename("USD-LTC"),
    Stream.source(list(bitfinex_tron['close']), dtype="float").rename("USD-TRX")
)

USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
USDT: Instrument = Instrument('USDT', 8, 'Tether')
BTC: Instrument = Instrument('BTC', 8, 'Bitcoin')
ETH: Instrument = Instrument('ETH', 8, 'Ethereum')
ADA: Instrument = Instrument('ADA', 8, 'Cardano')
SOL: Instrument = Instrument('SOL', 8, 'Solana')
LTC: Instrument = Instrument('LTC', 8, 'Litecoin')
TRX: Instrument = Instrument('TRX', 8, 'TRON')

portfolio = Portfolio(USD, [
    Wallet(bitfinex, 10000 * USD),
    Wallet(bitfinex, 0 * BTC),
    Wallet(bitfinex, 0 * ETH),
    Wallet(bitfinex, 0 * ADA),
    # Wallet(bitfinex, 0 * SOL),
    # Wallet(bitfinex, 0 * LTC),
    # Wallet(bitfinex, 0 * TRX),
])

print(f'Portfolio net worth: {portfolio.base_balance}')

from pandas import DataFrame, Series
from numpy import arange
import matplotlib.pyplot as plt

from tensortrade.env.default import create
from tensortrade.feed.core import DataFeed, Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.env.generic import Renderer
from tensortrade.env.default.rewards import PBR
from tensortrade.env.default.actions import BSH

from stable_baselines3 import A2C

from gym import register as gym_register, make as gym_make, Env


class PositionChangeChart(Renderer):

    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env: Env, **kwargs):
        history = DataFrame(env.observer.renderer_history)

        actions = list(history.action)
        p = list(history.price)

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


def create_btc_env(config):
    p: Stream = Stream.source(list(bitfinex_btc['close']), dtype="float").rename("USD-BTC")

    bitfinex = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash = Wallet(bitfinex, 100000 * USD)
    asset = Wallet(bitfinex, 0 * BTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    # Creating a comprehensive DataFeed with more meaningful features
    feed = DataFeed([
        p,
        
        # For Bitcoin (BTC)
        Stream.source(list(bitfinex_btc['LOGRET_16']), dtype="float").rename("log_return:/USD-BTC"),
        Stream.source(list(bitfinex_btc['RSI_14']), dtype="float").rename("rsi:/USD-BTC"),
        Stream.source(list(bitfinex_btc['MACD_12_26_9']), dtype="float").rename("macd:/USD-BTC"),
        Stream.source(list(bitfinex_btc['ROC_10']), dtype="float").rename("roc:/USD-BTC"),
        Stream.source(list(bitfinex_btc['BBL_20_2.0']), dtype="float").rename("bbl:/USD-BTC"),
        Stream.source(list(bitfinex_btc['BBM_20_2.0']), dtype="float").rename("bbm:/USD-BTC"),
        Stream.source(list(bitfinex_btc['BBB_20_2.0']), dtype="float").rename("bbb:/USD-BTC"),
        Stream.source(list(bitfinex_btc['BBP_20_2.0']), dtype="float").rename("bbp:/USD-BTC"),
        Stream.source(list(bitfinex_btc['OBV']), dtype="float").rename("obv:/USD-BTC"),
        Stream.source(list(bitfinex_btc['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-BTC"),
        Stream.source(list(bitfinex_btc['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-BTC"),

        # For Ethereum (ETH)
        Stream.source(list(bitfinex_eth['LOGRET_16']), dtype="float").rename("log_return:/USD-ETH"),
        Stream.source(list(bitfinex_eth['RSI_14']), dtype="float").rename("rsi:/USD-ETH"),

        # For Cardano (ADA)
        Stream.source(list(bitfinex_ada['LOGRET_16']), dtype="float").rename("log_return:/USD-ADA"),
        Stream.source(list(bitfinex_ada['RSI_14']), dtype="float").rename("rsi:/USD-ADA"),

        # # For Solana (SOL)
        Stream.source(list(bitfinex_sol['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-SOL"),
        Stream.source(list(bitfinex_sol['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-SOL"),

        # # For Litecoin (LTC)
        Stream.source(list(bitfinex_ltc['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-LTC"),

        # # For TRON (TRX)
        Stream.source(list(bitfinex_tron['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-TRX"),
        Stream.source(list(bitfinex_tron['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-TRX"),
    ])

    reward_scheme = PBR(price=p)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(bitfinex_btc['close'], dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    return create(feed=feed,
                  portfolio=portfolio,
                  action_scheme=action_scheme,
                  reward_scheme=reward_scheme,
                  renderer_feed=renderer_feed,
                  renderer=PositionChangeChart(),
                  window_size=config["window_size"],
                  max_allowed_loss=0.6)


def create_ada_env(config):
    p: Stream = Stream.source(list(bitfinex_ada['close']), dtype="float").rename("USD-ADA")

    bitfinex = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash = Wallet(bitfinex, 100000 * USD)
    asset = Wallet(bitfinex, 0 * ADA)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    # Creating a comprehensive DataFeed with more meaningful features
    feed = DataFeed([
        p,
        
        # For Bitcoin (BTC)
        Stream.source(list(bitfinex_btc['LOGRET_16']), dtype="float").rename("log_return:/USD-BTC"),
        Stream.source(list(bitfinex_btc['RSI_14']), dtype="float").rename("rsi:/USD-BTC"),

        # For Ethereum (ETH)
        Stream.source(list(bitfinex_eth['LOGRET_16']), dtype="float").rename("log_return:/USD-ETH"),
        Stream.source(list(bitfinex_eth['RSI_14']), dtype="float").rename("rsi:/USD-ETH"),

        # For Cardano (ADA)
        Stream.source(list(bitfinex_ada['LOGRET_16']), dtype="float").rename("log_return:/USD-ADA"),
        Stream.source(list(bitfinex_ada['RSI_14']), dtype="float").rename("rsi:/USD-ADA"),
        Stream.source(list(bitfinex_ada['MACD_12_26_9']), dtype="float").rename("macd:/USD-ADA"),
        Stream.source(list(bitfinex_ada['ROC_10']), dtype="float").rename("roc:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBL_20_2.0']), dtype="float").rename("bbl:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBM_20_2.0']), dtype="float").rename("bbm:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBB_20_2.0']), dtype="float").rename("bbb:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBP_20_2.0']), dtype="float").rename("bbp:/USD-ADA"),
        Stream.source(list(bitfinex_ada['OBV']), dtype="float").rename("obv:/USD-ADA"),
        Stream.source(list(bitfinex_ada['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-ADA"),
        Stream.source(list(bitfinex_ada['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-ADA"),

        # # For Solana (SOL)
        Stream.source(list(bitfinex_sol['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-SOL"),
        Stream.source(list(bitfinex_sol['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-SOL"),

        # # For Litecoin (LTC)
        Stream.source(list(bitfinex_ltc['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-LTC"),

        # # For TRON (TRX)
        Stream.source(list(bitfinex_tron['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-TRX"),
        Stream.source(list(bitfinex_tron['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-TRX"),
    ])

    reward_scheme = PBR(price=p)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(bitfinex_ada['close'], dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    return create(feed=feed,
                  portfolio=portfolio,
                  action_scheme=action_scheme,
                  reward_scheme=reward_scheme,
                  renderer_feed=renderer_feed,
                  renderer=PositionChangeChart(),
                  window_size=config["window_size"],
                  max_allowed_loss=0.9)


def create_eth_env(config):
    p: Stream = Stream.source(list(bitfinex_eth['close']), dtype="float").rename("USD-ETH")

    bitfinex = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash = Wallet(bitfinex, 100000 * USD)
    asset = Wallet(bitfinex, 0 * ETH)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    # Creating a comprehensive DataFeed with more meaningful features
    feed = DataFeed([
        p,
        
        # # For Bitcoin (BTC)
        Stream.source(list(bitfinex_btc['LOGRET_16']), dtype="float").rename("log_return:/USD-BTC"),
        Stream.source(list(bitfinex_btc['RSI_14']), dtype="float").rename("rsi:/USD-BTC"),

        # For Ethereum (ETH)
        Stream.source(list(bitfinex_eth['LOGRET_16']), dtype="float").rename("log_return:/USD-ETH"),
        Stream.source(list(bitfinex_eth['RSI_14']), dtype="float").rename("rsi:/USD-ETH"),
        Stream.source(list(bitfinex_eth['MACD_12_26_9']), dtype="float").rename("macd:/USD-ETH"),
        Stream.source(list(bitfinex_eth['ROC_10']), dtype="float").rename("roc:/USD-ETH"),
        Stream.source(list(bitfinex_eth['BBL_20_2.0']), dtype="float").rename("bbl:/USD-ETH"),
        Stream.source(list(bitfinex_eth['BBM_20_2.0']), dtype="float").rename("bbm:/USD-ETH"),
        Stream.source(list(bitfinex_eth['BBB_20_2.0']), dtype="float").rename("bbb:/USD-ETH"),
        Stream.source(list(bitfinex_eth['BBP_20_2.0']), dtype="float").rename("bbp:/USD-ETH"),
        Stream.source(list(bitfinex_eth['OBV']), dtype="float").rename("obv:/USD-ETH"),
        Stream.source(list(bitfinex_eth['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-ETH"),
        Stream.source(list(bitfinex_eth['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-ETH"),

        # For Cardano (ADA)
        Stream.source(list(bitfinex_ada['LOGRET_16']), dtype="float").rename("log_return:/USD-ADA"),
        Stream.source(list(bitfinex_ada['RSI_14']), dtype="float").rename("rsi:/USD-ADA"),

        # For Solana (SOL)
        Stream.source(list(bitfinex_sol['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-SOL"),
        Stream.source(list(bitfinex_sol['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-SOL"),

        # For Litecoin (LTC)
        Stream.source(list(bitfinex_ltc['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-LTC"),

        # For TRON (TRX)
        Stream.source(list(bitfinex_tron['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-TRX"),
        Stream.source(list(bitfinex_tron['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-TRX"),
    ])

    reward_scheme = PBR(price=p)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(bitfinex_eth['close'], dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    return create(feed=feed,
                  portfolio=portfolio,
                  action_scheme=action_scheme,
                  reward_scheme=reward_scheme,
                  renderer_feed=renderer_feed,
                  renderer=PositionChangeChart(),
                  window_size=config["window_size"],
                  max_allowed_loss=0.9)

gym_register("CryptoTradingEnv-ADA-v0", create_ada_env)
gym_register("CryptoTradingEnv-ETH-v0", create_eth_env)
gym_register("CryptoTradingEnv-BTC-v0", create_btc_env)


if __name__ == "__main__":
    envs: List[str] = ["CryptoTradingEnv-BTC-v0", "CryptoTradingEnv-ETH-v0", "CryptoTradingEnv-ADA-v0"]
    
    for env_name in envs:
        # Create environment
        env = gym_make(env_name, config={"window_size": 100})

        model = A2C("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=20000, log_interval=4)

        episodes: int = 100
        
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)

            if done:
                net_worth: float = DataFrame().from_dict(env.action_scheme.portfolio.performance, orient='index').iloc[-1]['net_worth']

                best_net_worth = net_worth
                print(f"Best net worth: {net_worth}")
                env.render(env_name = env_name)
