import numpy as np
from collections import defaultdict

class NumberPredictor:
    def __init__(self, df, has_bonus=True, max_main=52, main_count=6):
        self.df = df
        self.has_bonus = has_bonus
        self.max_main = max_main
        self.main_count = main_count
    
    def generate_prediction(self, group_pattern):
        """Generate actual numbers based on predicted group composition."""
        # group_pattern can be (g1,g2,g3,g4) or (g1,g2,g3,g4,bonus_group)
        main_counts = group_pattern[:4]
        main_numbers = self._sample_main_numbers(main_counts)
        
        bonus = None
        if self.has_bonus and len(group_pattern) > 4:
            bonus_group = group_pattern[4]
            bonus = self._sample_bonus(bonus_group)
        
        return {'main': sorted(main_numbers), 'bonus': bonus}
    
    def _sample_main_numbers(self, counts):
        """Sample main numbers respecting group counts and historical frequencies."""
        # For small max_main (Sportstake), ignore group counts and sample uniformly from 1..max_main
        if self.max_main <= 10:
            pool = list(range(1, self.max_main + 1))
            selected = np.random.choice(pool, size=self.main_count, replace=False)
            return sorted(selected)
        
        # Build candidate pool per group from last 100 draws
        candidates = {'g1': [], 'g2': [], 'g3': [], 'g4': []}
        recent_df = self.df.tail(100)
        for _, row in recent_df.iterrows():
            for n in row['main_numbers']:
                if n <= 10:
                    candidates['g1'].append(n)
                elif n <= 20:
                    candidates['g2'].append(n)
                elif n <= 30:
                    candidates['g3'].append(n)
                else:
                    candidates['g4'].append(n)
        
        # For each group, sample numbers weighted by their frequency in recent draws
        selected = []
        group_names = ['g1', 'g2', 'g3', 'g4']
        for g_name, cnt in zip(group_names, counts):
            pool = candidates[g_name]
            if not pool:
                # Fallback: generate numbers within group range
                if g_name == 'g1':
                    pool = list(range(1, 11))
                elif g_name == 'g2':
                    pool = list(range(11, 21))
                elif g_name == 'g3':
                    pool = list(range(21, 31))
                else:
                    pool = list(range(31, self.max_main + 1))
            # Weight by popularity
            unique_vals, freq = np.unique(pool, return_counts=True)
            probs = freq / freq.sum()
            chosen = np.random.choice(unique_vals, size=cnt, replace=False, p=probs)
            selected.extend(chosen)
        
        return sorted(selected)
    
    def _sample_bonus(self, bonus_group):
        """Sample bonus ball from low (1-10) or high (11-20) based on historical frequency."""
        low_pool = list(range(1, 11))
        high_pool = list(range(11, 21))
        pool = low_pool if bonus_group == 'low' else high_pool
        
        # Get historical bonus balls from last 100 draws
        hist_bonus = self.df['bonus'].dropna().tail(100).values
        hist_in_pool = [b for b in hist_bonus if b in pool]
        if hist_in_pool:
            # Weight by frequency
            unique, counts = np.unique(hist_in_pool, return_counts=True)
            probs = counts / counts.sum()
            return np.random.choice(unique, p=probs)
        else:
            return np.random.choice(pool)
