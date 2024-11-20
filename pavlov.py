# Built-in imports
from typing import Dict, Tuple

# Third-party imports
from pandas import DataFrame
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO, QRDQN
from gym import make as gym_make, register as gym_register
from tensortrade.oms.instruments import Instrument

from envs.CryptoEnv import create_crypto_train_env, create_crypto_test_env, create_btc_env


def single_security_tests(total_timesteps: int):
    # Register environments
    environment_pairs: Dict[str, Tuple[str, str]] = {
        'BTC': ('CryptoTradingTrainEnv-BTC-v0', 'CryptoTradingTestEnv-BTC-v0'),
        'ETH': ('CryptoTradingTrainEnv-ETH-v0', 'CryptoTradingTestEnv-ETH-v0'),
        'ADA': ('CryptoTradingTrainEnv-ADA-v0', 'CryptoTradingTestEnv-ADA-v0'),
        'SOL': ('CryptoTradingTrainEnv-SOL-v0', 'CryptoTradingTestEnv-SOL-v0'),
        'LTC': ('CryptoTradingTrainEnv-LTC-v0', 'CryptoTradingTestEnv-LTC-v0'),
        'TRX': ('CryptoTradingTrainEnv-TRX-v0', 'CryptoTradingTestEnv-TRX-v')
    }
    
    for asset_name, (train_env_name, test_env_name) in environment_pairs.items():
        gym_register(train_env_name, create_crypto_train_env)
        gym_register(test_env_name, create_crypto_test_env)
        
        cash_instrument: Instrument = Instrument('USD', 2)
        asset_instrument: Instrument = Instrument(asset_name, 8)
        env_config: dict = {
            "window_size": 40,
            'cash': cash_instrument,
            'assets': [asset_instrument]
        }
        
        print(f"Training model for {train_env_name}")
        # Create environment
        train_env = gym_make(train_env_name, config=env_config)

        # model: PPO = PPO("MlpPolicy", train_env, verbose=1)
        model: DQN = DQN("MlpPolicy", train_env, verbose=1)
        model.learn(total_timesteps=total_timesteps, log_interval=4)

        print(f"Testing model for {test_env_name}")
        
        train_env.close()

        test_env = gym_make(test_env_name, config=env_config)

        obs = test_env.reset()
        # model.set_env(train_env)
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = test_env.step(action)

            if done:
                net_worth: float = DataFrame().from_dict(test_env.action_scheme.portfolio.performance, orient='index').iloc[-1]['net_worth']

                print(f"Test net worth: {net_worth}")
                train_env.render(env_name=test_env_name,
                                 asset_name=asset_name,
                                 cash_name='USD')
        test_env.close()
        

def multi_asset_security_test(total_timesteps: int):
    train_env_name = 'MultiCryptoTradingTrainEnv-v0'
    test_env_name = 'MultiCryptoTradingTestEnv-v0'
    
    gym_register(train_env_name, create_crypto_train_env)
    gym_register(test_env_name, create_crypto_test_env)
    
    print(f"Training model for {train_env_name}")
    # Create environment
    train_env = gym_make(train_env_name, 
                         config={"window_size": 40,
                                 'cash': Instrument('USD', 2),
                                 'assets': [Instrument('BTC', 8),
                                            Instrument('ETH', 8),
                                            Instrument('ADA', 8),
                                            Instrument('SOL', 8),
                                            Instrument('LTC', 8),
                                            Instrument('TRX', 8)]})

    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=4)

    print(f"Testing model for {test_env_name}")
    
    test_env = gym_make(test_env_name, 
                        config={"window_size": 40,
                                'cash': Instrument('USD', 2),
                                'assets': [Instrument('BTC', 8),
                                           Instrument('ETH', 8),
                                           Instrument('ADA', 8),
                                           Instrument('SOL', 8),
                                           Instrument('LTC', 8),
                                           Instrument('TRX', 8)]})
    obs = test_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = test_env.step(action)
        model.train()

        if done:
            net_worth: float = DataFrame().from_dict(test_env.action_scheme.portfolio.performance, orient='index').iloc[-1]['net_worth']

            print(f"Test net worth: {net_worth}")
            test_env.render(env_name=test_env_name)


def btc_from_the_future():
    env_name: str = 'BTCCryptoTradingTrainEnv'
    gym_register(env_name, create_btc_env)
    
    env = gym_make(env_name,
                   config={"window_size": 40,
                           'cash': Instrument('USD', 2),
                           'assets': [Instrument('BTC', 8),
                                      Instrument('ETH', 8),
                                      Instrument('ADA', 8),
                                      Instrument('SOL', 8),
                                      Instrument('LTC', 8),
                                      Instrument('TRX', 8)]})

    model = PPO("MlpPolicy", env, verbose=1)
    # model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000 * 2.5, log_interval=4)
    
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            net_worth: float = DataFrame().from_dict(env.action_scheme.portfolio.performance, orient='index').iloc[-1]['net_worth']

            print(f"Test net worth: {net_worth}")
            env.render(env_name=env_name,
                       asset_name='BTC',
                       cash_name='USD')
    env.close()
    
    
if __name__ == "__main__":
    btc_from_the_future()
    single_security_tests(1e6)
    multi_asset_security_test(10000 * 2.5)