# Built-in imports
from os.path import abspath, join, dirname
from sys import path as sys_path
from typing import List

# Add the parent directory to the sys.path for local imports
sys_path.append(abspath(join(dirname(__file__), '..')))

# Third-party imports
from pandas import DataFrame, read_csv, DateOffset
import pandas_ta as ta
from numpy import array
from tensortrade.env.default import create
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
from tensortrade.feed.core import DataFeed, Stream, DataFeed
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import Instrument
# Local imports
from renderers.position_change import PositionChangeChart, MultiAssetPositionChangeChart
from rewards.pnl import PnLRewardScheme, MutliAssetPnLRewardScheme
from rewards.pbr import MultiAssetPBR
from actions.multi_asset_bsh import MultiAssetBSH


DATA_DIRECTORY: str = "/home/durzo/Pavlov/data/bitfinex"

bitfinex_btc: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_btc.csv'), parse_dates=['date'])
bitfinex_eth: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_eth.csv'), parse_dates=['date'])
bitfinex_ada: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_ada.csv'), parse_dates=['date'])
bitfinex_sol: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_sol.csv'), parse_dates=['date'])
bitfinex_ltc: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_ltc.csv'), parse_dates=['date'])
bitfinex_tron: DataFrame = read_csv(join(DATA_DIRECTORY, 'bitfinex_tron.csv'), parse_dates=['date'])

dfs: List[DataFrame] = [bitfinex_btc, bitfinex_eth, bitfinex_ada,
                        bitfinex_sol, bitfinex_ltc, bitfinex_tron]

train_dfs: List[DataFrame] = []
test_dfs: List[DataFrame] = []

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
    train_dfs.append(df[df['date'] < split_date])
    test_dfs.append(df[df['date'] >= split_date])

def create_crypto_env(config):
    assert 'window_size' in config, "Window size is required for Crypto environment"
    assert 'assets' in config, "Assets, in the form of instruments, are required for Crypto environment"
    assert 'cash' in config, "Cash, in the form of an instrument, is required for Crypto environment"
    
    cash: Instrument = config['cash']
    assets: List[Instrument] = config['assets']
    
    

    p: Stream = Stream.source(list(bitfinex_btc['close']), dtype="float").rename(f"{cash.name}-BTC")

    bitfinex: Exchange = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash_wallet: Wallet = Wallet(bitfinex, 10000 * USD)
    btc_wallet: Wallet = Wallet(bitfinex, 0 * BTC)

    portfolio = Portfolio(USD, [
        cash_wallet,
        btc_wallet
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

    reward_scheme = PnLRewardScheme(price=p,
                                    cash_wallet=cash_wallet,
                                    asset_wallet=btc_wallet)

    action_scheme = BSH(
        cash=cash_wallet,
        asset=btc_wallet
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


def create_btc_env(config):
    USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
    BTC: Instrument = Instrument('BTC', 8, 'Bitcoin')

    p: Stream = Stream.source(list(bitfinex_btc['close']), dtype="float").rename("USD-BTC")

    bitfinex: Exchange = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash_wallet: Wallet = Wallet(bitfinex, 10000 * USD)
    btc_wallet: Wallet = Wallet(bitfinex, 0 * BTC)

    portfolio = Portfolio(USD, [
        cash_wallet,
        btc_wallet
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

    reward_scheme = PnLRewardScheme(price=p,
                                    cash_wallet=cash_wallet,
                                    asset_wallet=btc_wallet)

    action_scheme = BSH(
        cash=cash_wallet,
        asset=btc_wallet
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
    USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
    ADA: Instrument = Instrument('ADA', 8, 'Cardano')

    p: Stream = Stream.source(list(bitfinex_ada['close']), dtype="float").rename("USD-ADA")

    bitfinex: Exchange = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash_wallet: Wallet = Wallet(bitfinex, 10000 * USD)
    asset = Wallet(bitfinex, 0 * ADA)

    portfolio = Portfolio(USD, [
        cash_wallet,
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

    # reward_scheme = PnLRewardScheme(price=p,
    #                                 cash_wallet=cash_wallet,
    #                                 asset_wallet=asset)

    reward_scheme = PBR(price=p)

    action_scheme = BSH(
        cash=cash_wallet,
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
    USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
    ETH: Instrument = Instrument('ETH', 8, 'Ethereum')

    p: Stream = Stream.source(list(bitfinex_eth['close']), dtype="float").rename("USD-ETH")

    bitfinex: Exchange = Exchange("bitfinex", service=execute_order)(
        p
    )

    cash_wallet: Wallet = Wallet(bitfinex, 10000 * USD)
    eth_wallet = Wallet(bitfinex, 0 * ETH)

    portfolio = Portfolio(USD, [
        cash_wallet,
        eth_wallet
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
        cash=cash_wallet,
        asset=eth_wallet
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


def multi_asset_crypto_env(config):
    USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
    BTC: Instrument = Instrument('BTC', 8, 'Bitcoin')
    ETH: Instrument = Instrument('ETH', 8, 'Ethereum')
    ADA: Instrument = Instrument('ADA', 8, 'Cardano')
    SOL: Instrument = Instrument('SOL', 8, 'Solana')
    LTC: Instrument = Instrument('LTC', 8, 'Litecoin')
    TRX: Instrument = Instrument('TRX', 8, 'TRON')

    btc_price: Stream = Stream.source(list(bitfinex_btc['close']), dtype="float").rename("USD-BTC")
    eth_price: Stream = Stream.source(list(bitfinex_eth['close']), dtype="float").rename("USD-ETH")
    ada_price: Stream = Stream.source(list(bitfinex_ada['close']), dtype="float").rename("USD-ADA")
    sol_price: Stream = Stream.source(list(bitfinex_sol['close']), dtype="float").rename("USD-SOL")
    ltc_price: Stream = Stream.source(list(bitfinex_ltc['close']), dtype="float").rename("USD-LTC")
    tron_price: Stream = Stream.source(list(bitfinex_tron['close']), dtype="float").rename("USD-TRX")
    
    bitfinex: Exchange = Exchange("bitfinex", service=execute_order)(
        btc_price,
        eth_price,
        ada_price,
        sol_price,
        ltc_price,
        tron_price
    )    

    cash_wallet: Wallet = Wallet(bitfinex, 10000 * USD)
    btc_wallet: Wallet = Wallet(bitfinex, 0 * BTC)
    eth_wallet: Wallet = Wallet(bitfinex, 0 * ETH)
    ada_wallet: Wallet = Wallet(bitfinex, 0 * ADA)
    sol_wallet: Wallet = Wallet(bitfinex, 0 * SOL)
    ltc_wallet: Wallet = Wallet(bitfinex, 0 * LTC)
    tron_wallet: Wallet = Wallet(bitfinex, 0 * TRX)
    
    portfolio = Portfolio(USD, [
        cash_wallet,
        btc_wallet,
        eth_wallet,
        ada_wallet,
        sol_wallet,
        ltc_wallet,
        tron_wallet
    ])
    
    # Creating a comprehensive DataFeed with more meaningful features
    feed = DataFeed([
        # USD prices for each asset
        btc_price,
        eth_price,
        ada_price,
        sol_price,
        ltc_price,
        tron_price,
        
        # # For Bitcoin (BTC)
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
        Stream.source(list(bitfinex_ada['MACD_12_26_9']), dtype="float").rename("macd:/USD-ADA"),
        Stream.source(list(bitfinex_ada['ROC_10']), dtype="float").rename("roc:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBL_20_2.0']), dtype="float").rename("bbl:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBM_20_2.0']), dtype="float").rename("bbm:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBB_20_2.0']), dtype="float").rename("bbb:/USD-ADA"),
        Stream.source(list(bitfinex_ada['BBP_20_2.0']), dtype="float").rename("bbp:/USD-ADA"),
        Stream.source(list(bitfinex_ada['OBV']), dtype="float").rename("obv:/USD-ADA"),
        Stream.source(list(bitfinex_ada['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-ADA"),
        Stream.source(list(bitfinex_ada['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-ADA"),

        # For Solana (SOL)
        Stream.source(list(bitfinex_sol['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-SOL"),
        Stream.source(list(bitfinex_sol['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-SOL"),
        Stream.source(list(bitfinex_sol['MACD_12_26_9']), dtype="float").rename("macd:/USD-SOL"),
        Stream.source(list(bitfinex_sol['ROC_10']), dtype="float").rename("roc:/USD-SOL"),
        Stream.source(list(bitfinex_sol['BBL_20_2.0']), dtype="float").rename("bbl:/USD-SOL"),
        Stream.source(list(bitfinex_sol['BBM_20_2.0']), dtype="float").rename("bbm:/USD-SOL"),
        Stream.source(list(bitfinex_sol['BBB_20_2.0']), dtype="float").rename("bbb:/USD-SOL"),
        Stream.source(list(bitfinex_sol['BBP_20_2.0']), dtype="float").rename("bbp:/USD-SOL"),
        Stream.source(list(bitfinex_sol['OBV']), dtype="float").rename("obv:/USD-SOL"),
        Stream.source(list(bitfinex_sol['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-SOL"),
        Stream.source(list(bitfinex_sol['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-SOL"),

        # For Litecoin (LTC)
        Stream.source(list(bitfinex_ltc['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['MACD_12_26_9']), dtype="float").rename("macd:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['ROC_10']), dtype="float").rename("roc:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['BBL_20_2.0']), dtype="float").rename("bbl:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['BBM_20_2.0']), dtype="float").rename("bbm:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['BBB_20_2.0']), dtype="float").rename("bbb:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['BBP_20_2.0']), dtype="float").rename("bbp:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['OBV']), dtype="float").rename("obv:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-LTC"),
        Stream.source(list(bitfinex_ltc['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-LTC"),

        # For TRON (TRX)
        Stream.source(list(bitfinex_tron['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-TRX"),
        Stream.source(list(bitfinex_tron['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-TRX"),
        Stream.source(list(bitfinex_ada['MACD_12_26_9']), dtype="float").rename("macd:/USD-TRX"),
        Stream.source(list(bitfinex_tron['ROC_10']), dtype="float").rename("roc:/USD-TRX"),
        Stream.source(list(bitfinex_tron['BBL_20_2.0']), dtype="float").rename("bbl:/USD-TRX"),
        Stream.source(list(bitfinex_tron['BBM_20_2.0']), dtype="float").rename("bbm:/USD-TRX"),
        Stream.source(list(bitfinex_tron['BBB_20_2.0']), dtype="float").rename("bbb:/USD-TRX"),
        Stream.source(list(bitfinex_tron['BBP_20_2.0']), dtype="float").rename("bbp:/USD-TRX"),
        Stream.source(list(bitfinex_tron['OBV']), dtype="float").rename("obv:/USD-ADATRX"),
        Stream.source(list(bitfinex_tron['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-TRX"),
        Stream.source(list(bitfinex_tron['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-TRX"),
    ])

    reward_scheme = MultiAssetPBR(price_streams=[btc_price, eth_price, 
                                                 ada_price, sol_price,
                                                 ltc_price, tron_price])

    action_scheme = MultiAssetBSH(
        cash_wallet=cash_wallet,
        asset_wallets=[btc_wallet, eth_wallet, ada_wallet,
                       sol_wallet, ltc_wallet, tron_wallet]
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(bitfinex_eth['close'], dtype="float").rename("eth_price"),
        Stream.source(bitfinex_btc['close'], dtype="float").rename("btc_price"),
        Stream.source(bitfinex_ada['close'], dtype="float").rename("ada_price"),
        Stream.source(bitfinex_sol['close'], dtype="float").rename("sol_price"),
        Stream.source(bitfinex_ltc['close'], dtype="float").rename("ltc_price"),
        Stream.source(bitfinex_tron['close'], dtype="float").rename("tron_price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    return create(feed=feed,
                  portfolio=portfolio,
                  action_scheme=action_scheme,
                  reward_scheme=reward_scheme,
                  renderer_feed=renderer_feed,
                  renderer=MultiAssetPositionChangeChart(),
                  window_size=config["window_size"],
                  max_allowed_loss=0.9)
