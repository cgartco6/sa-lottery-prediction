import pandas as pd
from pathlib import Path

class DataLoader:
    @staticmethod
    def load_all(raw_dir='data/raw/', config_lotteries=None):
        """
        Load all CSV files from raw_dir.
        config_lotteries: dict from settings.yaml, keys are lottery names.
        Expected filenames: {lottery_name}.csv
        """
        if config_lotteries is None:
            raise ValueError("config_lotteries must be provided")
        
        data = {}
        for name in config_lotteries.keys():
            filename = f"{name}.csv"
            path = Path(raw_dir) / filename
            if not path.exists():
                print(f"Warning: {path} not found. Skipping {name}.")
                continue
            
            df = pd.read_csv(path)
            
            # Convert main_numbers column from string to list of ints
            # Expected format: "1,2,3,4,5" or "[1,2,3,4,5]"
            if 'main_numbers' in df.columns:
                df['main_numbers'] = df['main_numbers'].apply(
                    lambda x: [int(i) for i in str(x).strip('[]').split(',')]
                )
            else:
                # If column is missing, try to find columns like number1, number2,...
                number_cols = [c for c in df.columns if c.startswith('number') or c.startswith('outcome')]
                if number_cols:
                    df['main_numbers'] = df[number_cols].values.tolist()
                    df['main_numbers'] = df['main_numbers'].apply(lambda row: [int(x) for x in row])
                else:
                    raise KeyError(f"No valid main_numbers column found in {filename}")
            
            # Handle bonus column
            if 'bonus' not in df.columns:
                df['bonus'] = None
            else:
                df['bonus'] = df['bonus'].fillna(None)
            
            data[name] = df
        
        return data
