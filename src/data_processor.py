import pandas as pd

class DataProcessor:
    @staticmethod
    def add_group_columns(df, has_bonus, max_main):
        """
        Adds group columns (g1,g2,g3,g4) for main numbers.
        If max_main <= 10, treat all numbers as g1 (single group).
        If has_bonus, adds bonus_group (low/high).
        """
        if max_main > 10:
            # Standard 4 groups based on SA lotteries
            def count_groups(nums):
                counts = {'g1': 0, 'g2': 0, 'g3': 0, 'g4': 0}
                for n in nums:
                    if n <= 10:
                        counts['g1'] += 1
                    elif n <= 20:
                        counts['g2'] += 1
                    elif n <= 30:
                        counts['g3'] += 1
                    else:
                        counts['g4'] += 1
                return pd.Series(counts)
        else:
            # For small max_main (e.g. Sportstake), treat all as g1
            def count_groups(nums):
                return pd.Series({'g1': len(nums), 'g2': 0, 'g3': 0, 'g4': 0})
        
        df[['g1', 'g2', 'g3', 'g4']] = df['main_numbers'].apply(count_groups)
        
        if has_bonus and 'bonus' in df.columns:
            df['bonus_group'] = df['bonus'].apply(
                lambda x: 'low' if pd.notna(x) and x <= 10 else 'high' if pd.notna(x) else None
            )
        
        return df
