# Built-in imports
from os.path import abspath, join, dirname
from sys import path as sys_path
from typing import List

# Third-party imports
from pandas import DataFrame, read_csv, DateOffset
from numpy import array
from stable_baselines3 import A2C
from gym import make as gym_make, register as gym_register

from envs.CryptoEnv import create_btc_env, create_eth_env, create_ada_env, multi_asset_crypto_env


if __name__ == "__main__":
    # gym_register("CryptoTradingEnv-ADA-v0", create_ada_env)
    # gym_register("CryptoTradingEnv-ETH-v0", create_eth_env)
    # gym_register("CryptoTradingEnv-BTC-v0", create_btc_env)
    gym_register("CryptoTradingEnv-Multi-v0", multi_asset_crypto_env)

    envs: List[str] = ["CryptoTradingEnv-Multi-v0", 
                    #    "CryptoTradingEnv-BTC-v0", 
                    #    "CryptoTradingEnv-ETH-v0", 
                    #    "CryptoTradingEnv-ADA-v0"
                       ]
    
    for env_name in envs:
        print(f"Training model for {env_name}")
        # Create environment
        env = gym_make(env_name, config={"window_size": 100})

        model = A2C("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1e5, log_interval=4)
        
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

    DATA_DIRECTORY: str = "/home/durzo/Pavlov/data/bitfinex"

    bitfinex_btc: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_btc.csv'), parse_dates=['date'])
    bitfinex_eth: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_eth.csv'), parse_dates=['date'])
    bitfinex_ada: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_ada.csv'), parse_dates=['date'])
    bitfinex_sol: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_sol.csv'), parse_dates=['date'])
    bitfinex_ltc: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_ltc.csv'), parse_dates=['date'])
    bitfinex_tron: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_tron.csv'), parse_dates=['date'])

    dfs: List[DataFrame] = [bitfinex_btc, bitfinex_eth, bitfinex_ada,
                            bitfinex_sol, bitfinex_ltc, bitfinex_tron]

    for df in dfs:
        df.ta.log_return(append=True, length=16)
        df.ta.rsi(append=True, length=14)
        df.ta.macd(append=True, fast=12, slow=26)
        df.ta.roc(append=True, length=10)
        df.ta.bbands(append=True, length=20, std=2)
        df.ta.obv(append=True)
        df.ta.stoch(append=True, k=14, d=3, smooth_k=3)
        df.dropna(inplace=True)
        
        # Cut the data to only include training data
        # Calculate the split date for the last 3 months
        split_date = df['date'].max() - DateOffset(months=3)

        # Split into training and test sets
        df = df[df['date'] >= split_date]
        
    
