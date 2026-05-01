from collections import Counter
import numpy as np

class GroupAnalyzer:
    def __init__(self, df, has_bonus=True, use_last_n=None):
        """
        df: DataFrame with group columns (g1,g2,g3,g4) and optionally bonus_group.
        has_bonus: if True, include bonus_group in pattern.
        use_last_n: if int, only use last N draws; else use all.
        """
        if use_last_n is not None:
            self.df = df.tail(use_last_n)
        else:
            self.df = df
        self.has_bonus = has_bonus
        self.pattern_counts = Counter()
        self._build_patterns()
    
    def _build_patterns(self):
        for _, row in self.df.iterrows():
            pattern = (row['g1'], row['g2'], row['g3'], row['g4'])
            if self.has_bonus and 'bonus_group' in row and pd.notna(row['bonus_group']):
                pattern = pattern + (row['bonus_group'],)
            self.pattern_counts[pattern] += 1
    
    def most_common_patterns(self, top_n=5):
        return self.pattern_counts.most_common(top_n)
    
    def predict_next_groups(self):
        """Return a predicted group pattern weighted by historical frequencies."""
        if not self.pattern_counts:
            # Fallback: return default pattern (e.g., all numbers spread)
            if self.has_bonus:
                return (2, 1, 2, 1, 'low')
            else:
                return (2, 1, 2, 1)
        
        patterns, freqs = zip(*self.pattern_counts.items())
        probs = np.array(freqs) / sum(freqs)
        predicted = patterns[np.random.choice(len(patterns), p=probs)]
        return predicted
