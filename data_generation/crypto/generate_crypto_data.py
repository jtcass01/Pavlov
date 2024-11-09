
__author__ = "Jacob Taylor Cassady"
__email__ = "jcassad1@jh.edu"

from tensortrade.data.cdd import CryptoDataDownload
from pandas import DataFrame


if __name__ == "__main__":
    cdd: CryptoDataDownload = CryptoDataDownload()
    print(f"Downloading bitcoin data...")
    bitfinex_btc: DataFrame = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    bitfinex_btc.to_csv('bitfinex_btc.csv')
    print(f"Downloaded and saved bitcoin data.")

    print(f"Downloading ethereum data...")
    bitfinex_eth: DataFrame = cdd.fetch("Bitfinex", "USD", "ETH", "1h")
    bitfinex_eth.to_csv('bitfinex_eth.csv')
    print(f"Downloaded and saved ethereum data.")

    print(f"Downloading cardano data...")
    bitfinex_ada: DataFrame = cdd.fetch("Bitfinex", "USD", "ADA", "1h")
    bitfinex_ada.to_csv('bitfinex_ada.csv')
    print(f"Downloaded and saved cardano data.")

    print(f"Downloading solana data...")
    bitfinex_sol: DataFrame = cdd.fetch("Bitfinex", "USD", "SOL", "1h")
    bitfinex_sol.to_csv('bitfinex_sol.csv')
    print(f"Downloaded and saved solana data.")

    print(f"Downloading litecoin data...")
    bitfinex_ltc: DataFrame = cdd.fetch("Bitfinex", "USD", "LTC", "1h")
    bitfinex_ltc.to_csv('bitfinex_ltc.csv')
    print(f"Downloaded and saved litecoin data.")

    print(f"Downloading TRON data...")
    bitfinex_tron: DataFrame = cdd.fetch("Bitfinex", "USD", "TRX", "1h")
    bitfinex_tron.to_csv('bitfinex_tron.csv')
    print(f"Downloaded and saved TRON data.")
