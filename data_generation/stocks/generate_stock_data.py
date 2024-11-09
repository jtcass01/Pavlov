from argparse import ArgumentParser
from yfinance import Ticker
import pandas_ta as ta

if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description='Generate training and evaluation data for a given ticker.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--train_start_date', type=str, default='2023-02-09', help='Training start date')
    parser.add_argument('--train_end_date', type=str, default='2023-09-30', help='Training end date')
    parser.add_argument('--eval_start_date', type=str, default='2023-10-01', help='Evaluation start date')
    parser.add_argument('--eval_end_date', type=str, default='2023-11-12', help='Evaluation end date')

    args = parser.parse_args()

    # Assign arguments to variables
    ticker: str = args.ticker
    train_start_date: str = args.train_start_date
    train_end_date: str = args.train_end_date
    eval_start_date: str = args.eval_start_date
    evaL_end_date: str = args.eval_end_date

    # Fetch data and process
    yf_ticker = Ticker(ticker=ticker)

    df_training = yf_ticker.history(start=train_start_date, end=train_end_date, interval='60m')
    df_training.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    df_training["Volume"] = df_training["Volume"].astype(int)
    df_training.ta.log_return(append=True, length=16)
    df_training.ta.rsi(append=True, length=14)
    df_training.ta.macd(append=True, fast=12, slow=26)
    df_training.to_csv('training.csv', index=True)

    df_evaluation = yf_ticker.history(start=eval_start_date, end=evaL_end_date, interval='60m')
    df_evaluation.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    df_evaluation["Volume"] = df_evaluation["Volume"].astype(int)
    df_evaluation.ta.log_return(append=True, length=16)
    df_evaluation.ta.rsi(append=True, length=14)
    df_evaluation.ta.macd(append=True, fast=12, slow=26)
    df_evaluation.to_csv('evaluation.csv', index=True)
