from os.path import join
from typing import Dict, List

from numpy import exp
from pandas import DataFrame, read_csv, DateOffset, Series
from matplotlib.pyplot import show, legend, plot, title, subplots

BANK_INTEREST_RATE: float = 0.04

if __name__ == "__main__":
    DATA_DIRECTORY: str = "/home/durzo/Pavlov/data/bitfinex"
    RESULTS_DIRECTORY: str = "/home/durzo/Pavlov/models"
    TEST_MONTHS: int = 5
    models: List[str] = ['single_security_tests', 'single_security_risk_tests']
    assets: List[str] = ['BTC', 'ETH', 'ADA', 'SOL', 'LTC', 'TRX']
    
    raw_data: Dict[str, DataFrame] = {}
    for asset in assets:
        raw_data[asset] = read_csv(join(DATA_DIRECTORY, f'bitfinex_{asset.lower()}.csv'), parse_dates=['date'])

    train_dfs: Dict[str, DataFrame] = {}
    test_dfs: Dict[str, DataFrame] = {}

    for asset, df in raw_data.items():
        # Cut the data to only include training data
        # Calculate the split date for the last 4 months
        split_date = df['date'].max() - DateOffset(months=5)

        # Split into training and test sets
        train_dfs[asset] = df[df['date'] < split_date]
        test_dfs[asset] = df[df['date'] >= split_date]

    # Get best test results for each asset
    test_results: Dict[str, dict] = {}
    for model in models:
        model_directory: str = join(RESULTS_DIRECTORY, model)
        test_results[model] = {}
        
        for asset in assets:
            asset_directory: str = join(model_directory, asset)
            test_results[model][asset] = read_csv(join(asset_directory, f'{asset}_performance.csv'))
    
    figure, axes = subplots(nrows=2, ncols=3, figsize=(12, 6))

    # Plot the close prices for each asset in a 3x2 grid
    for asset_index, (asset, df) in enumerate(test_dfs.items()):
        # Create a space to store model agnostic data
        test_results[asset] = {}
        
        initial_usd: float = test_results[model][asset]['net_worth'].iloc[0]
        asset_price: Series = test_results[model][asset][f'bitfinex:/USD-{asset}']
        initial_price: float = asset_price.iloc[0]

        test_results_size: int = len(test_results[model][asset])        
        dates: Series = df['date'].reset_index(drop=True).iloc[-test_results_size:]
        time_in_years = dates.apply(lambda x: (x - dates.iloc[0]).total_seconds() / (60 * 60 * 24 * 365))
        savings_account = time_in_years.apply(lambda t: initial_usd * exp(BANK_INTEREST_RATE * t))
        
        # How much of the asset can be HODLed
        hodl_amount: float = initial_usd / initial_price
        test_results[asset]['hodl_worth'] = hodl_amount*asset_price
        
        plot_axes = axes[asset_index // 3, asset_index % 3]
        plot_axes.plot(dates[:len(test_results['single_security_tests'][asset]['net_worth'])], 
                       test_results['single_security_tests'][asset]['net_worth'], 
                       label=f'BSH Net Worth', 
                       color='blue')
        plot_axes.plot(dates[:len(test_results['single_security_risk_tests'][asset]['net_worth'])], 
                       test_results['single_security_risk_tests'][asset]['net_worth'], 
                       label=f'Managed Risk Net Worth', 
                       color='green')
        plot_axes.plot(dates[:len(test_results[asset]['hodl_worth'])], 
                       test_results[asset]['hodl_worth'], 
                       label=f'HODL Net Worth', 
                       color='red')
        plot_axes.plot(dates[:len(savings_account)], 
                       savings_account, 
                       label=f'Savings Account Net Worth', 
                       color='black')
        plot_axes.title.set_text(f'{asset} Trading Strategies')
        plot_axes.grid(True, which='both', linestyle='--')
        plot_axes.set_facecolor('#e6e6e6')
        plot_axes.legend()
        
    show()

    # Create objects for keeping track of totals across models
    total_net_worth: Dict[str, Series] = {}
    for asset in assets:
        # BSH
        if 'single_security_tests' not in total_net_worth:
            total_net_worth['single_security_tests'] = test_results['single_security_tests'][asset]['net_worth']
        else:
            total_net_worth['single_security_tests'] += test_results['single_security_tests'][asset]['net_worth']
        
        # Managed Risk
        if 'single_security_risk_tests' not in total_net_worth:
            total_net_worth['single_security_risk_tests'] = test_results['single_security_risk_tests'][asset]['net_worth']
        else:
            total_net_worth['single_security_risk_tests'] += test_results['single_security_risk_tests'][asset]['net_worth']

        # Saving Account
        test_results_size: int = max([len(asset_model) for asset_model in test_results[model].values()])       
        dates: Series = df['date'].reset_index(drop=True).iloc[-test_results_size:]
        time_in_years = dates.apply(lambda x: (x - dates.iloc[0]).total_seconds() / (60 * 60 * 24 * 365))
        savings_account = time_in_years.apply(lambda t: initial_usd * exp(BANK_INTEREST_RATE * t))
        
        if 'hodl' not in total_net_worth:
            total_net_worth['hodl'] = test_results[asset]['hodl_worth']
        else:
            total_net_worth['hodl'] += test_results[asset]['hodl_worth']
            
        if 'savings' not in total_net_worth:
            total_net_worth['savings'] = savings_account
        else:
            total_net_worth['savings'] += savings_account

    figure, axes = subplots(nrows=1, ncols=1, figsize=(12, 6))
    axes.plot(dates[:len(total_net_worth['single_security_tests'])], 
                total_net_worth['single_security_tests'], 
                label=f'BSH Net Worth',
                color='blue')
    axes.plot(dates[:len(total_net_worth['single_security_risk_tests'])], 
                total_net_worth['single_security_risk_tests'], 
                label=f'Managed Risk Net Worth',
                color='green')
    axes.plot(dates[:len(total_net_worth['hodl'])], 
              total_net_worth['hodl'], 
              label=f'HODL Net Worth',
              color='red')
    axes.plot(dates[:len(total_net_worth['savings'])], 
              total_net_worth['savings'], 
              label=f'Savings Account Net Worth',
              color='black')    
    axes.title.set_text(f'All Trading Strategies')
    axes.grid(True, which='both', linestyle='--')
    axes.set_facecolor('#e6e6e6')
    axes.legend()
    show()
    