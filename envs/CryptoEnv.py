# Built-in imports
from os.path import abspath, join, dirname
from sys import path as sys_path
from typing import List, Dict

# Add the parent directory to the sys.path for local imports
sys_path.append(abspath(join(dirname(__file__), '..')))

# Third-party imports
from pandas import DataFrame, read_csv, DateOffset
import pandas_ta as ta
from numpy import array
from tensortrade.env.default import create, TradingEnv
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
from rewards.vap import VAP
from actions.multi_asset_bsh import MultiAssetBSH
from actions.psas import PSAS


DATA_DIRECTORY: str = "/home/durzo/Pavlov/data/bitfinex"

raw_data: Dict[str, DataFrame] = {
    'BTC': read_csv(join(DATA_DIRECTORY, 'bitfinex_btc.csv'), parse_dates=['date']),
    'ETH': read_csv(join(DATA_DIRECTORY, 'bitfinex_eth.csv'), parse_dates=['date']),
    'ADA': read_csv(join(DATA_DIRECTORY, 'bitfinex_ada.csv'), parse_dates=['date']),
    'SOL': read_csv(join(DATA_DIRECTORY, 'bitfinex_sol.csv'), parse_dates=['date']),
    'LTC': read_csv(join(DATA_DIRECTORY, 'bitfinex_ltc.csv'), parse_dates=['date']),
    'TRX': read_csv(join(DATA_DIRECTORY, 'bitfinex_tron.csv'), parse_dates=['date'])
}

train_dfs: Dict[str, DataFrame] = {}
test_dfs: Dict[str, DataFrame] = {}

for df_key, df in raw_data.items():
    df.ta.rsi(append=True, length=14)
    df.ta.cci(append=True, length=30)
    df.ta.adx(append=True, length=30)
    df.ta.sma(append=True, length=30)
    df.ta.sma(append=True, length=60)
    df.ta.macd(append=True, fast=12, slow=26)
    df.ta.bbands(append=True, length=20, std=2)
    df.ta.atr(append=True)

    df.ta.log_return(append=True, length=16)
    df.ta.roc(append=True, length=10)
    df.ta.obv(append=True)
    df.ta.stoch(append=True, k=14, d=3, smooth_k=3)

    df.dropna(inplace=True)
    
    # Cut the data to only include training data
    # Calculate the split date for the last 4 months
    split_date = df['date'].max() - DateOffset(months=3)

    # Split into training and test sets
    train_dfs[df_key] = df[df['date'] < split_date]
    print(f"Size of training data for {df_key}: {len(train_dfs[df_key])}")
    test_dfs[df_key] = df[df['date'] >= split_date]
    print(f"Size of test data for {df_key}: {len(test_dfs[df_key])}")
    
    print(df.head())

def create_crypto_train_env(config) -> TradingEnv:
    return create_crypto_env(config, train_dfs)

def create_crypto_test_env(config) -> TradingEnv:
    return create_crypto_env(config, test_dfs)

def create_crypto_env(config, df: DataFrame) -> TradingEnv:
    assert 'window_size' in config, "Window size is required for Crypto environment"
    assert 'assets' in config, "Assets, in the form of instruments, are required for Crypto environment"
    assert 'cash' in config, "Cash, in the form of an instrument, is required for Crypto environment"
    
    cash: Instrument = config['cash']
    assets: List[Instrument] = config['assets']

    price_streams: List[Stream] = []
    renderer_streams: List[Stream] = []
    for asset in assets:
        price_stream: Stream = Stream.source(list(df[str(asset)]['close']), dtype="float").rename(f"{cash}-{asset}")
        price_streams.append(price_stream)
        renderer_streams.append(price_stream)

    bitfinex: Exchange = Exchange("bitfinex", service=execute_order)(
        *price_streams
    )

    cash_wallet: Wallet = Wallet(bitfinex, 1e4 * cash)
    wallets: List[Wallet] = [cash_wallet]
    for asset in assets:
        wallets.append(Wallet(bitfinex, 0 * asset))

    portfolio = Portfolio(cash, wallets)

    # Creating a comprehensive DataFeed with more meaningful features
    assets_included: list = [str(asset) for asset in assets]
    additional_asset_targets: list = [asset for asset in raw_data.keys() if asset not in assets_included]
    for asset in assets:
        price_streams.append(Stream.source(list(df[str(asset)]['RSI_14']), dtype="float").rename(f"rsi:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['CCI_30_0.015']), dtype="float").rename(f"cci:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['ADX_30']), dtype="float").rename(f"cci:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['DMP_30']), dtype="float").rename(f"cci:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['DMN_30']), dtype="float").rename(f"cci:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['SMA_30']), dtype="float").rename(f"cci:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['SMA_60']), dtype="float").rename(f"cci:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['MACD_12_26_9']), dtype="float").rename(f"macd:/{cash}-{asset}"))

        price_streams.append(Stream.source(list(df[str(asset)]['LOGRET_16']), dtype="float").rename(f"log_return:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['ROC_10']), dtype="float").rename(f"roc:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['OBV']), dtype="float").rename(f"obv:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['STOCHk_14_3_3']), dtype="float").rename(f"stochk:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['STOCHd_14_3_3']), dtype="float").rename(f"stockd:/{cash}-{asset}"))

        price_streams.append(Stream.source(list(df[str(asset)]['BBL_20_2.0']), dtype="float").rename(f"bbl:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['BBM_20_2.0']), dtype="float").rename(f"bbm:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['BBB_20_2.0']), dtype="float").rename(f"bbb:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['BBP_20_2.0']), dtype="float").rename(f"bbp:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['ATRr_14']), dtype="float").rename(f"bbp:/{cash}-{asset}"))
        
    for asset in additional_asset_targets:
        price_streams.append(Stream.source(list(df[str(asset)]['RSI_14']), dtype="float").rename(f"rsi:/{cash}-{asset}"))
        price_streams.append(Stream.source(list(df[str(asset)]['MACD_12_26_9']), dtype="float").rename(f"macd:/{cash}-{asset}"))

    # for asset in assets:
    #     price_streams.append(Stream.source(list(df[str(asset)]['RSI_14']), dtype="float").rename(f"rsi:/{cash}-{asset}"))
    #     price_streams.append(Stream.source(list(df[str(asset)]['MACD_12_26_9']), dtype="float").rename(f"macd:/{cash}-{asset}"))
    #     price_streams.append(Stream.source(list(df[str(asset)]['BBL_20_2.0']), dtype="float").rename(f"bbl:/{cash}-{asset}"))
    #     price_streams.append(Stream.source(list(df[str(asset)]['BBM_20_2.0']), dtype="float").rename(f"bbm:/{cash}-{asset}"))
    #     price_streams.append(Stream.source(list(df[str(asset)]['BBB_20_2.0']), dtype="float").rename(f"bbb:/{cash}-{asset}"))
    #     price_streams.append(Stream.source(list(df[str(asset)]['BBP_20_2.0']), dtype="float").rename(f"bbp:/{cash}-{asset}"))
    
    # for asset in additional_asset_targets:
    #     price_streams.append(Stream.source(list(df[str(asset)]['LOGRET_16']), dtype="float").rename(f"log_return:/{cash}-{asset}"))
    #     price_streams.append(Stream.source(list(df[str(asset)]['RSI_14']), dtype="float").rename(f"rsi:/{cash}-{asset}"))
    
    print(len(bitfinex._price_streams))
    
    feed = DataFeed(price_streams)

    if len(assets) == 1:
        
        reward_scheme: PBR = PBR(price=price_streams[0])
        # reward_scheme: PnLRewardScheme = PnLRewardScheme(price=price_streams[0],
        #                                      cash_wallet=cash_wallet,
        #                                      asset_wallet=wallets[1])
        # reward_scheme: VAP = VAP(price=price_streams[0],
        #                          window_size=config["window_size"])

        action_scheme: BSH = BSH(cash=cash_wallet, asset=wallets[1]).attach(reward_scheme)
        # action_scheme: PSAS = PSAS(cash=cash_wallet, asset=wallets[1], max_scale=0.9)

        renderer_streams.append(Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action"))
        renderer_streams.append(Stream.source(list(train_dfs[str(asset)]['close']), dtype="float").rename(f"price"))
        # Rename the price stream to "price" for the renderer
        renderer_feed = DataFeed(renderer_streams)

        return create(feed=feed,
                      portfolio=portfolio,
                      action_scheme=action_scheme,
                      reward_scheme=reward_scheme,
                      renderer_feed=renderer_feed,
                      renderer=PositionChangeChart(),
                      window_size=config["window_size"],
                      max_allowed_loss=0.6)
        
    reward_scheme: MultiAssetPBR = MultiAssetPBR(price_streams)
    action_scheme: MultiAssetBSH = MultiAssetBSH(cash_wallet=cash_wallet, asset_wallets=wallets[1:]).attach(reward_scheme)

    renderer_streams.append(Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action"))
    renderer_feed = DataFeed(renderer_streams)

    return create(feed=feed,
                  portfolio=portfolio,
                  action_scheme=action_scheme,
                  reward_scheme=reward_scheme,
                  renderer_feed=renderer_feed,
                  renderer=MultiAssetPositionChangeChart(),
                  window_size=config["window_size"],
                  max_allowed_loss=0.9)

def create_btc_train_env(config):
    USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
    BTC: Instrument = Instrument('BTC', 8, 'Bitcoin')

    p: Stream = Stream.source(list(train_dfs['BTC']['close']), dtype="float").rename("USD-BTC")

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
        Stream.source(list(train_dfs['BTC']['LOGRET_16']), dtype="float").rename("log_return:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['RSI_14']), dtype="float").rename("rsi:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['MACD_12_26_9']), dtype="float").rename("macd:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['ROC_10']), dtype="float").rename("roc:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['BBL_20_2.0']), dtype="float").rename("bbl:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['BBM_20_2.0']), dtype="float").rename("bbm:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['BBB_20_2.0']), dtype="float").rename("bbb:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['BBP_20_2.0']), dtype="float").rename("bbp:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['OBV']), dtype="float").rename("obv:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-BTC"),
        Stream.source(list(train_dfs['BTC']['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-BTC"),

        # For Ethereum (ETH)
        Stream.source(list(train_dfs['ETH']['LOGRET_16']), dtype="float").rename("log_return:/USD-ETH"),
        Stream.source(list(train_dfs['ETH']['RSI_14']), dtype="float").rename("rsi:/USD-ETH"),

        # For Cardano (ADA)
        Stream.source(list(train_dfs['ADA']['LOGRET_16']), dtype="float").rename("log_return:/USD-ADA"),
        Stream.source(list(train_dfs['ADA']['RSI_14']), dtype="float").rename("rsi:/USD-ADA"),

        # # For Solana (SOL)
        Stream.source(list(train_dfs['SOL']['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-SOL"),
        Stream.source(list(train_dfs['SOL']['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-SOL"),

        # # For Litecoin (LTC)
        Stream.source(list(train_dfs['LTC']['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-LTC"),
        Stream.source(list(train_dfs['LTC']['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-LTC"),

        # # For TRON (TRX)
        Stream.source(list(train_dfs['TRX']['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-TRX"),
        Stream.source(list(train_dfs['TRX']['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-TRX"),
    ])
    
    reward_scheme = PBR(price=p)
    # reward_scheme: VAP = VAP(price=p,
    #                          window_size=config["window_size"])

    action_scheme = BSH(
        cash=cash_wallet,
        asset=btc_wallet
    ).attach(reward_scheme)
    # action_scheme: PSAS = PSAS(cash=cash_wallet, asset=btc_wallet, max_scale=0.9)

    renderer_feed = DataFeed([
        Stream.source(train_dfs['BTC']['close'], dtype="float").rename("bitfinex:/USD-BTC"),
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

def create_btc_test_env(config):
    USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
    BTC: Instrument = Instrument('BTC', 8, 'Bitcoin')

    p: Stream = Stream.source(list(test_dfs['BTC']['close']), dtype="float").rename("USD-BTC")

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
        Stream.source(list(test_dfs['BTC']['LOGRET_16']), dtype="float").rename("log_return:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['RSI_14']), dtype="float").rename("rsi:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['MACD_12_26_9']), dtype="float").rename("macd:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['ROC_10']), dtype="float").rename("roc:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['BBL_20_2.0']), dtype="float").rename("bbl:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['BBM_20_2.0']), dtype="float").rename("bbm:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['BBB_20_2.0']), dtype="float").rename("bbb:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['BBP_20_2.0']), dtype="float").rename("bbp:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['OBV']), dtype="float").rename("obv:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-BTC"),
        Stream.source(list(test_dfs['BTC']['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-BTC"),

        # For Ethereum (ETH)
        Stream.source(list(test_dfs['ETH']['LOGRET_16']), dtype="float").rename("log_return:/USD-ETH"),
        Stream.source(list(test_dfs['ETH']['RSI_14']), dtype="float").rename("rsi:/USD-ETH"),

        # For Cardano (ADA)
        Stream.source(list(test_dfs['ADA']['LOGRET_16']), dtype="float").rename("log_return:/USD-ADA"),
        Stream.source(list(test_dfs['ADA']['RSI_14']), dtype="float").rename("rsi:/USD-ADA"),

        # # For Solana (SOL)
        Stream.source(list(test_dfs['SOL']['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-SOL"),
        Stream.source(list(test_dfs['SOL']['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-SOL"),

        # # For Litecoin (LTC)
        Stream.source(list(test_dfs['LTC']['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-LTC"),
        Stream.source(list(test_dfs['LTC']['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-LTC"),

        # # For TRON (TRX)
        Stream.source(list(test_dfs['TRX']['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-TRX"),
        Stream.source(list(test_dfs['TRX']['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-TRX"),
    ])
    
    reward_scheme = PBR(price=p)
    # reward_scheme: VAP = VAP(price=p,
    #                          window_size=config["window_size"])

    action_scheme = BSH(
        cash=cash_wallet,
        asset=btc_wallet
    ).attach(reward_scheme)
    # action_scheme: PSAS = PSAS(cash=cash_wallet, asset=btc_wallet, max_scale=0.9)

    renderer_feed = DataFeed([
        Stream.source(test_dfs['BTC']['close'], dtype="float").rename("bitfinex:/USD-BTC"),
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

# def multi_asset_crypto_env(config):
#     USD: Instrument = Instrument('USD', 2, 'U.S. Dollar')
#     BTC: Instrument = Instrument('BTC', 8, 'Bitcoin')
#     ETH: Instrument = Instrument('ETH', 8, 'Ethereum')
#     ADA: Instrument = Instrument('ADA', 8, 'Cardano')
#     SOL: Instrument = Instrument('SOL', 8, 'Solana')
#     LTC: Instrument = Instrument('LTC', 8, 'Litecoin')
#     TRX: Instrument = Instrument('TRX', 8, 'TRON')

#     btc_price: Stream = Stream.source(list(bitfinex_btc['close']), dtype="float").rename("USD-BTC")
#     eth_price: Stream = Stream.source(list(bitfinex_eth['close']), dtype="float").rename("USD-ETH")
#     ada_price: Stream = Stream.source(list(bitfinex_ada['close']), dtype="float").rename("USD-ADA")
#     sol_price: Stream = Stream.source(list(bitfinex_sol['close']), dtype="float").rename("USD-SOL")
#     ltc_price: Stream = Stream.source(list(bitfinex_ltc['close']), dtype="float").rename("USD-LTC")
#     tron_price: Stream = Stream.source(list(bitfinex_tron['close']), dtype="float").rename("USD-TRX")
    
#     bitfinex: Exchange = Exchange("bitfinex", service=execute_order)(
#         btc_price,
#         eth_price,
#         ada_price,
#         sol_price,
#         ltc_price,
#         tron_price
#     )    

#     cash_wallet: Wallet = Wallet(bitfinex, 10000 * USD)
#     btc_wallet: Wallet = Wallet(bitfinex, 0 * BTC)
#     eth_wallet: Wallet = Wallet(bitfinex, 0 * ETH)
#     ada_wallet: Wallet = Wallet(bitfinex, 0 * ADA)
#     sol_wallet: Wallet = Wallet(bitfinex, 0 * SOL)
#     ltc_wallet: Wallet = Wallet(bitfinex, 0 * LTC)
#     tron_wallet: Wallet = Wallet(bitfinex, 0 * TRX)
    
#     portfolio = Portfolio(USD, [
#         cash_wallet,
#         btc_wallet,
#         eth_wallet,
#         ada_wallet,
#         sol_wallet,
#         ltc_wallet,
#         tron_wallet
#     ])
    
#     # Creating a comprehensive DataFeed with more meaningful features
#     feed = DataFeed([
#         # USD prices for each asset
#         btc_price,
#         eth_price,
#         ada_price,
#         sol_price,
#         ltc_price,
#         tron_price,
        
#         # # For Bitcoin (BTC)
#         Stream.source(list(bitfinex_btc['LOGRET_16']), dtype="float").rename("log_return:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['RSI_14']), dtype="float").rename("rsi:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['MACD_12_26_9']), dtype="float").rename("macd:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['ROC_10']), dtype="float").rename("roc:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['BBL_20_2.0']), dtype="float").rename("bbl:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['BBM_20_2.0']), dtype="float").rename("bbm:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['BBB_20_2.0']), dtype="float").rename("bbb:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['BBP_20_2.0']), dtype="float").rename("bbp:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['OBV']), dtype="float").rename("obv:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-BTC"),
#         Stream.source(list(bitfinex_btc['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-BTC"),

#         # For Ethereum (ETH)
#         Stream.source(list(bitfinex_eth['LOGRET_16']), dtype="float").rename("log_return:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['RSI_14']), dtype="float").rename("rsi:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['MACD_12_26_9']), dtype="float").rename("macd:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['ROC_10']), dtype="float").rename("roc:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['BBL_20_2.0']), dtype="float").rename("bbl:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['BBM_20_2.0']), dtype="float").rename("bbm:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['BBB_20_2.0']), dtype="float").rename("bbb:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['BBP_20_2.0']), dtype="float").rename("bbp:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['OBV']), dtype="float").rename("obv:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-ETH"),
#         Stream.source(list(bitfinex_eth['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-ETH"),

#         # For Cardano (ADA)
#         Stream.source(list(bitfinex_ada['LOGRET_16']), dtype="float").rename("log_return:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['RSI_14']), dtype="float").rename("rsi:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['MACD_12_26_9']), dtype="float").rename("macd:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['ROC_10']), dtype="float").rename("roc:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['BBL_20_2.0']), dtype="float").rename("bbl:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['BBM_20_2.0']), dtype="float").rename("bbm:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['BBB_20_2.0']), dtype="float").rename("bbb:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['BBP_20_2.0']), dtype="float").rename("bbp:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['OBV']), dtype="float").rename("obv:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-ADA"),
#         Stream.source(list(bitfinex_ada['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-ADA"),

#         # For Solana (SOL)
#         Stream.source(list(bitfinex_sol['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['MACD_12_26_9']), dtype="float").rename("macd:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['ROC_10']), dtype="float").rename("roc:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['BBL_20_2.0']), dtype="float").rename("bbl:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['BBM_20_2.0']), dtype="float").rename("bbm:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['BBB_20_2.0']), dtype="float").rename("bbb:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['BBP_20_2.0']), dtype="float").rename("bbp:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['OBV']), dtype="float").rename("obv:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-SOL"),
#         Stream.source(list(bitfinex_sol['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-SOL"),

#         # For Litecoin (LTC)
#         Stream.source(list(bitfinex_ltc['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['MACD_12_26_9']), dtype="float").rename("macd:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['ROC_10']), dtype="float").rename("roc:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['BBL_20_2.0']), dtype="float").rename("bbl:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['BBM_20_2.0']), dtype="float").rename("bbm:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['BBB_20_2.0']), dtype="float").rename("bbb:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['BBP_20_2.0']), dtype="float").rename("bbp:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['OBV']), dtype="float").rename("obv:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-LTC"),
#         Stream.source(list(bitfinex_ltc['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-LTC"),

#         # For TRON (TRX)
#         Stream.source(list(bitfinex_tron['LOGRET_16'][-100:]), dtype="float").rename("log_return:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['RSI_14'][-100:]), dtype="float").rename("rsi:/USD-TRX"),
#         Stream.source(list(bitfinex_ada['MACD_12_26_9']), dtype="float").rename("macd:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['ROC_10']), dtype="float").rename("roc:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['BBL_20_2.0']), dtype="float").rename("bbl:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['BBM_20_2.0']), dtype="float").rename("bbm:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['BBB_20_2.0']), dtype="float").rename("bbb:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['BBP_20_2.0']), dtype="float").rename("bbp:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['OBV']), dtype="float").rename("obv:/USD-ADATRX"),
#         Stream.source(list(bitfinex_tron['STOCHk_14_3_3']), dtype="float").rename("stochk:/USD-TRX"),
#         Stream.source(list(bitfinex_tron['STOCHd_14_3_3']), dtype="float").rename("stockd:/USD-TRX"),
#     ])

#     reward_scheme = MultiAssetPBR(price_streams=[btc_price, eth_price, 
#                                                  ada_price, sol_price,
#                                                  ltc_price, tron_price])

#     action_scheme = MultiAssetBSH(
#         cash_wallet=cash_wallet,
#         asset_wallets=[btc_wallet, eth_wallet, ada_wallet,
#                        sol_wallet, ltc_wallet, tron_wallet]
#     ).attach(reward_scheme)

#     renderer_feed = DataFeed([
#         Stream.source(bitfinex_eth['close'], dtype="float").rename("eth_price"),
#         Stream.source(bitfinex_btc['close'], dtype="float").rename("btc_price"),
#         Stream.source(bitfinex_ada['close'], dtype="float").rename("ada_price"),
#         Stream.source(bitfinex_sol['close'], dtype="float").rename("sol_price"),
#         Stream.source(bitfinex_ltc['close'], dtype="float").rename("ltc_price"),
#         Stream.source(bitfinex_tron['close'], dtype="float").rename("tron_price"),
#         Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
#     ])

#     return create(feed=feed,
#                   portfolio=portfolio,
#                   action_scheme=action_scheme,
#                   reward_scheme=reward_scheme,
#                   renderer_feed=renderer_feed,
#                   renderer=MultiAssetPositionChangeChart(),
#                   window_size=config["window_size"],
#                   max_allowed_loss=0.9)
