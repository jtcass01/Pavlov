# Built-in imports
from typing import Dict, Tuple, List
from random import choice
from os.path import join, exists, dirname

# Third-party imports
from pandas import DataFrame
from stable_baselines3 import A2C, PPO
from gym import make as gym_make, register as gym_register
from tensortrade.oms.instruments import Instrument

from envs.CryptoEnv import (create_crypto_train_env, 
                            create_crypto_test_env, 
                            create_btc_test_env, 
                            create_btc_train_env, 
                            create_crypto_risk_train_env, 
                            create_crypto_risk_test_env)
from util.model_logging import ModelLogging

MODELS_DIRECTORY: str = join(dirname(__file__), 'models')


def single_security_tests(tests: int = 50):
    # Register environments
    environment_pairs: Dict[str, Tuple[str, str]] = {
        'BTC': ('CryptoTradingTrainEnv-BTC-v0', 'CryptoTradingTestEnv-BTC-v0'),
        'ETH': ('CryptoTradingTrainEnv-ETH-v0', 'CryptoTradingTestEnv-ETH-v0'),
        'ADA': ('CryptoTradingTrainEnv-ADA-v0', 'CryptoTradingTestEnv-ADA-v0'),
        'SOL': ('CryptoTradingTrainEnv-SOL-v0', 'CryptoTradingTestEnv-SOL-v0'),
        'LTC': ('CryptoTradingTrainEnv-LTC-v0', 'CryptoTradingTestEnv-LTC-v0'),
        'TRX': ('CryptoTradingTrainEnv-TRX-v0', 'CryptoTradingTestEnv-TRX-v0')
    }
    
    # Setup model logging
    model_logging: ModelLogging = ModelLogging('single_security_tests', list(environment_pairs.keys()), MODELS_DIRECTORY)

    window_sizes: List[int] = [96, 72, 48, 24, 12]
    timesteps: List[int] = [500, 1000, 5000, 10000, 15000]

    for _ in range(tests):
        print(f"Test {_}:")
        for asset_name, (train_env_name, test_env_name) in environment_pairs.items():
            timestep: int = choice(timesteps)
            window_size: int = choice(window_sizes)

            gym_register(train_env_name, create_crypto_train_env)
            gym_register(test_env_name, create_crypto_test_env)
            
            cash_instrument: Instrument = Instrument('USD', 2)
            asset_instrument: Instrument = Instrument(asset_name, 8)
            env_config: dict = {
                "window_size": window_size,
                'cash': cash_instrument,
                'assets': [asset_instrument]
            }
            
            print(f"[{_}]\tTraining model for {train_env_name} with window size {window_size} and timesteps {timestep}")
            # Create environment
            train_env = gym_make(train_env_name, config=env_config)

            model: PPO = PPO("MlpPolicy", train_env, device='cpu')
            model.learn(total_timesteps=timestep, progress_bar=True)

            print(f"\tTesting model for {test_env_name}")
            
            train_env.close()

            test_env = gym_make(test_env_name, config=env_config)

            model.set_env(test_env)
            obs = test_env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = test_env.step(action)

                if done:
                    new_best = model_logging.log_model(
                        asset_name, model, test_env.action_scheme.portfolio.performance,
                        config={
                            'window_size': window_size,
                            'timesteps': timestep
                        })

                    if new_best:
                        test_env.render(env_name=test_env_name,
                                        asset_name=asset_name,
                                        cash_name='USD')
                    
            test_env.close()


def single_security_risk_tests(tests: int = 3):
    # Register environments
    environment_pairs: Dict[str, Tuple[str, str]] = {
        'BTC': ('CryptoTradingRiskTrainEnv-BTC-v0', 'CryptoTradingRiskTestEnv-BTC-v0'),
        'ETH': ('CryptoTradingRiskTrainEnv-ETH-v0', 'CryptoTradingRiskTestEnv-ETH-v0'),
        'ADA': ('CryptoTradingRiskTrainEnv-ADA-v0', 'CryptoTradingRiskTestEnv-ADA-v0'),
        'SOL': ('CryptoTradingRiskTrainEnv-SOL-v0', 'CryptoTradingRiskTestEnv-SOL-v0'),
        'LTC': ('CryptoTradingRiskTrainEnv-LTC-v0', 'CryptoTradingRiskTestEnv-LTC-v0'),
        'TRX': ('CryptoTradingRiskTrainEnv-TRX-v0', 'CryptoTradingRiskTestEnv-TRX-v0')
    }

    model_logging: ModelLogging = ModelLogging('single_security_risk_tests', list(environment_pairs.keys()), MODELS_DIRECTORY)
    
    window_sizes: List[int] = [96, 72, 48, 24, 12]
    timesteps: List[int] = [500, 1000, 5000, 10000, 15000]

    for _ in range(tests):
        print(f"Test {_}:")
        for asset_name, (train_env_name, test_env_name) in environment_pairs.items():
            timestep: int = choice(timesteps)
            window_size: int = choice(window_sizes)
            
            gym_register(train_env_name, create_crypto_risk_train_env)
            gym_register(test_env_name, create_crypto_risk_test_env)
            
            cash_instrument: Instrument = Instrument('USD', 2)
            asset_instrument: Instrument = Instrument(asset_name, 8)
            env_config: dict = {
                "window_size": window_size,
                'cash': cash_instrument,
                'assets': [asset_instrument]
            }
            
            print(f"[{_}]\tTraining model for {train_env_name} with window size {window_size} and timesteps {timestep}")

            # Create environment
            train_env = gym_make(train_env_name, config=env_config)

            model: PPO = PPO("MlpPolicy", train_env, device='cpu')
            model.learn(total_timesteps=timestep, progress_bar=True)

            print(f"[{_}]\tTesting model for {test_env_name}")
            
            train_env.close()

            test_env = gym_make(test_env_name, config=env_config)

            obs = test_env.reset()
            model.set_env(test_env)
            done = False
            _states = None
            while not done:
                action, _states = model.predict(obs, state=_states, deterministic=True)
                obs, rewards, done, info = test_env.step(action)

                if done:
                    model_logging.log_model(asset_name, model, test_env.action_scheme.portfolio.performance,
                                            config={
                                                'window_size': window_size,
                                                'timesteps': timestep
                                            })
                    
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
    gym_register(env_name, create_btc_train_env)
    
    env = gym_make(env_name,
                   config={"window_size": 40,
                           'cash': Instrument('USD', 2),
                           'assets': [Instrument('BTC', 8),
                                      Instrument('ETH', 8),
                                      Instrument('ADA', 8),
                                      Instrument('SOL', 8),
                                      Instrument('LTC', 8),
                                      Instrument('TRX', 8)]})

    model = A2C("MlpPolicy", env, verbose=1)
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


def btc_train_and_test():
    train_env_name = 'BTCCryptoTradingTrainEnv-v0'
    test_env_name = 'BTCCryptoTradingTestEnv-v0'
    
    env_config: dict = {
        "window_size": 40,
        'cash': Instrument('USD', 2),
        'assets': [Instrument('BTC', 8),
                   Instrument('ETH', 8),
                   Instrument('ADA', 8),
                   Instrument('SOL', 8),
                   Instrument('LTC', 8),
                   Instrument('TRX', 8)]
    }
    
    gym_register(train_env_name, create_btc_train_env)
    gym_register(test_env_name, create_btc_test_env)

    train_env = gym_make(train_env_name, config=env_config)

    model = A2C("MlpPolicy", train_env, verbose=1)
    # model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000, log_interval=4)

    test_env = gym_make(test_env_name, config=env_config)
    
    obs = test_env.reset()
    model.set_env(test_env)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)

        if done:
            net_worth: float = DataFrame().from_dict(test_env.action_scheme.portfolio.performance, orient='index').iloc[-1]['net_worth']

            print(f"Test net worth: {net_worth}")
            test_env.render(env_name=test_env,
                       asset_name='BTC',
                       cash_name='USD')
    test_env.close()
    
    
if __name__ == "__main__":
    # btc_train_and_test()
    # single_security_risk_tests(tests=int(1e4))
    single_security_tests(tests=int(1e4))
    # multi_asset_security_test(10000 * 2.5)