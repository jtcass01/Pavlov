# Built-in imports
from typing import List
from os import makedirs
from os.path import join, exists
from yaml import dump, load, FullLoader

# Third-party imports
from numpy import array
from pandas import DataFrame
from stable_baselines3.common.base_class import BaseAlgorithm
from matplotlib.pyplot import plot, savefig, legend, clf, title


class ModelLogging:
    def __init__(self, identifier: str, asset_names: List[str], models_directory: str) -> None:
        self._identifier: str = identifier
        self._asset_names: List[str] = asset_names
        self._best_net_worth: dict = {}
        
        # Create a directory for the model if it doesn't exist
        self._models_directory: str = models_directory
        self._model_directory: str = join(models_directory, identifier)
        if not exists(self._model_directory):
            makedirs(self._model_directory)

        # Create directories for each asset
        for asset_name in asset_names:
            asset_directory: str = join(self._model_directory, asset_name)
            if not exists(asset_directory):
                makedirs(asset_directory)
            
        # Check if existing best models exists
        self._best_models_path: str = join(self._model_directory, 'best_models.yaml')
        if exists(self._best_models_path):
            with open(self._best_models_path, 'r') as f:
                self._best_net_worth = load(f, Loader=FullLoader)
                print(f'Starting with best net worths: {self._best_net_worth}')
        else:
            for asset_name in asset_names:
                self._best_net_worth[asset_name] = -float('inf')
                
            self.log_best_net_worth()
            
    def log_best_net_worth(self):
        with open(self._best_models_path, 'w') as f:
            dump(self._best_net_worth, f)

    def log_model(self, asset_name: str, model: BaseAlgorithm, performance: dict,
                  config: dict) -> bool:
        is_new_best: bool = self.evaluate_model(asset_name, performance)
        if is_new_best:
            self.save_model(asset_name, model, performance, config)
        return is_new_best
            
    def evaluate_model(self, asset_name: str, performance: dict) -> bool:
        net_worth: array = DataFrame().from_dict(performance, orient='index')['net_worth'].to_numpy()
        shifted_net_worth: array = net_worth - net_worth[0]
        net_sum: float = float(sum(shifted_net_worth))
        return net_sum > self._best_net_worth[asset_name]

    def save_model(self, asset_name: str, model: BaseAlgorithm, performance: dict, config: dict) -> None:
        # Update the best net worth for the asset
        net_worth: DataFrame = DataFrame().from_dict(performance, orient='index')['net_worth'].to_numpy()
        shifted_net_worth: array = net_worth - net_worth[0]
        net_sum: float = float(sum(shifted_net_worth))
        self._best_net_worth[asset_name] = net_sum
        print(f"New best net worth for {asset_name}: {self._best_net_worth[asset_name]}")
        
        # Create a plot of the net worth
        clf()
        plot(net_worth, label=f"{asset_name} Net Worth")
        legend()
        title(f"{asset_name} Net Worth")
        net_worth_plot_file_path: str = join(self._model_directory, asset_name, f'{asset_name}_net_worth.png')
        savefig(net_worth_plot_file_path)

        # Save the performance data
        performance_df: DataFrame = DataFrame().from_dict(performance, orient='index')
        performance_file_path: str = join(self._model_directory, asset_name, f'{asset_name}_performance.csv')
        performance_df.to_csv(performance_file_path)

        # Create a plot of the performance
        clf()
        performance_plot_file_path: str = join(self._model_directory, asset_name, f'{asset_name}_performance.png')
        performance_df.plot()
        savefig(performance_plot_file_path)
        
        # Save the model
        model_file_path: str = join(self._model_directory, asset_name, f'{asset_name}_model')
        model.save(model_file_path)
        
        # Update the best models file
        self.log_best_net_worth()
        
        # Save the configuration
        config_file_path: str = join(self._model_directory, asset_name, f'{asset_name}_config.yaml')
        with open(config_file_path, 'w') as f: dump(config, f)
