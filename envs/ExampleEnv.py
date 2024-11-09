import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensortrade.env.default import create
from tensortrade.feed.core import DataFeed, Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.env.generic import Renderer
from tensortrade.env.default.rewards import PBR
from tensortrade.env.default.actions import BSH, ManagedRiskOrders

from stable_baselines3 import DQN

from gym import register as gym_register, make as gym_make

class PositionChangeChart(Renderer):

    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env, **kwargs):
        history = pd.DataFrame(env.observer.renderer_history)

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

        buy = pd.Series(buy)
        sell = pd.Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Performance")

        axs[0].plot(np.arange(len(p)), p, label="price", color=self.color)
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")

        performance_df = pd.DataFrame().from_dict(env.action_scheme.portfolio.performance, orient='index')
        performance_df.plot(ax=axs[1])
        axs[1].set_title("Net Worth")
        
        plt.savefig('trading_performance.png')
        plt.close()


def create_env(config):
    USD = Instrument("USD", 2, "U.S. Dollar")
    TTC = Instrument("TTC", 8, "TensorTrade Coin")

    x = np.arange(0, 2*np.pi, 2*np.pi / 1001)
    y = 50*np.sin(3*x) + 100

    x = np.arange(0, 2*np.pi, 2*np.pi / 1000)
    p = Stream.source(y, dtype="float").rename("USD-TTC")

    bitfinex = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash = Wallet(bitfinex, 100000 * USD)
    asset = Wallet(bitfinex, 0 * TTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    feed = DataFeed([
        p,
        p.rolling(window=10).mean().rename("fast"),
        p.rolling(window=50).mean().rename("medium"),
        p.rolling(window=100).mean().rename("slow"),
        p.log().diff().fillna(0).rename("lr")
    ])

    reward_scheme = PBR(price=p)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(y, dtype="float").rename("price"),
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


gym_register("CryptoTradingEnv-v0", create_env)


if __name__ == "__main__":
    # Create environment
    env = gym_make("CryptoTradingEnv-v0", config={"window_size": 25})

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("crypto_trading")
    
    del model
    
    model = DQN.load("crypto_trading")

    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            env.render()
            obs = env.reset()
