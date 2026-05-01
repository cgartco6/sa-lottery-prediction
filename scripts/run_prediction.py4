#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from collections import Counter

# -------------------------------
# Lottery definitions (embedded)
# -------------------------------
LOTTERY_CONFIG = {
    'daily': {'main_count': 5, 'max_main': 36, 'has_bonus': False},
    'lotto': {'main_count': 6, 'max_main': 52, 'has_bonus': True, 'bonus_max': 20},
    'lotto_plus1': {'main_count': 6, 'max_main': 52, 'has_bonus': True, 'bonus_max': 20},
    'lotto_plus2': {'main_count': 6, 'max_main': 52, 'has_bonus': True, 'bonus_max': 20},
    'powerball': {'main_count': 5, 'max_main': 50, 'has_bonus': True, 'bonus_max': 20},
    'powerball_plus': {'main_count': 5, 'max_main': 50, 'has_bonus': True, 'bonus_max': 20},
    'daily_lotto': {'main_count': 5, 'max_main': 36, 'has_bonus': False},
    'daily_lotto_plus1': {'main_count': 5, 'max_main': 36, 'has_bonus': False},
    'sportstake': {'main_count': 6, 'max_main': 4, 'has_bonus': False},
    'sportstake_8': {'main_count': 8, 'max_main': 4, 'has_bonus': False}
}

# -------------------------------
# Helper classes (simplified, no external imports except pandas/numpy)
# -------------------------------
class DataLoader:
    @staticmethod
    def load_all(raw_dir, lottery_names):
        data = {}
        for name in lottery_names:
            csv_path = Path(raw_dir) / f"{name}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found. Skipping {name}.")
                continue
            df = pd.read_csv(csv_path)
            # Find main numbers columns
            # Try 'main_numbers' first
            if 'main_numbers' in df.columns:
                df['main_numbers'] = df['main_numbers'].apply(
                    lambda x: [int(i) for i in str(x).strip('[]').split(',')]
                )
            else:
                # Look for columns like 'ball1','ball2',... or 'number1','number2',...
                ball_cols = [c for c in df.columns if c.startswith(('ball','number','outcome')) and c[-1].isdigit()]
                if ball_cols:
                    # Sort by number suffix
                    ball_cols.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                    df['main_numbers'] = df[ball_cols].values.tolist()
                    df['main_numbers'] = df['main_numbers'].apply(lambda row: [int(x) for x in row])
                else:
                    raise KeyError(f"No valid number columns found in {csv_path}")
            if 'bonus' not in df.columns:
                df['bonus'] = None
            else:
                df['bonus'] = df['bonus'].fillna(None)
            data[name] = df
        return data

class DataProcessor:
    @staticmethod
    def add_group_columns(df, has_bonus, max_main):
        if max_main > 10:
            def count_groups(nums):
                cnt = {'g1':0,'g2':0,'g3':0,'g4':0}
                for n in nums:
                    if n <= 10: cnt['g1']+=1
                    elif n <= 20: cnt['g2']+=1
                    elif n <= 30: cnt['g3']+=1
                    else: cnt['g4']+=1
                return pd.Series(cnt)
        else:
            def count_groups(nums):
                return pd.Series({'g1':len(nums),'g2':0,'g3':0,'g4':0})
        df[['g1','g2','g3','g4']] = df['main_numbers'].apply(count_groups)
        if has_bonus and 'bonus' in df.columns:
            df['bonus_group'] = df['bonus'].apply(lambda x: 'low' if pd.notna(x) and x<=10 else 'high' if pd.notna(x) else None)
        return df

class GroupAnalyzer:
    def __init__(self, df, has_bonus, use_last_n=None):
        self.df = df.tail(use_last_n) if use_last_n else df
        self.has_bonus = has_bonus
        self.pattern_counts = Counter()
        self._build()
    def _build(self):
        for _, row in self.df.iterrows():
            p = (row['g1'], row['g2'], row['g3'], row['g4'])
            if self.has_bonus and 'bonus_group' in row and pd.notna(row['bonus_group']):
                p = p + (row['bonus_group'],)
            self.pattern_counts[p] += 1
    def predict_next_groups(self):
        if not self.pattern_counts:
            return (2,1,2,1,'low') if self.has_bonus else (2,1,2,1)
        patterns, freqs = zip(*self.pattern_counts.items())
        probs = np.array(freqs)/sum(freqs)
        return patterns[np.random.choice(len(patterns), p=probs)]

class NumberPredictor:
    def __init__(self, df, has_bonus, max_main, main_count):
        self.df = df
        self.has_bonus = has_bonus
        self.max_main = max_main
        self.main_count = main_count
    def generate_prediction(self, group_pattern):
        main_counts = group_pattern[:4]
        main = self._sample_main(main_counts)
        bonus = None
        if self.has_bonus and len(group_pattern)>4:
            bonus = self._sample_bonus(group_pattern[4])
        return {'main': sorted(main), 'bonus': bonus}
    def _sample_main(self, counts):
        if self.max_main <= 10:
            pool = list(range(1,self.max_main+1))
            return sorted(np.random.choice(pool, size=self.main_count, replace=False))
        cand = {'g1':[],'g2':[],'g3':[],'g4':[]}
        recent = self.df.tail(100)
        for _,row in recent.iterrows():
            for n in row['main_numbers']:
                if n<=10: cand['g1'].append(n)
                elif n<=20: cand['g2'].append(n)
                elif n<=30: cand['g3'].append(n)
                else: cand['g4'].append(n)
        selected = []
        for g,cnt in zip(['g1','g2','g3','g4'], counts):
            pool = cand[g]
            if not pool:
                if g=='g1': pool=list(range(1,11))
                elif g=='g2': pool=list(range(11,21))
                elif g=='g3': pool=list(range(21,31))
                else: pool=list(range(31,self.max_main+1))
            uniq, freq = np.unique(pool, return_counts=True)
            probs = freq/freq.sum()
            selected.extend(np.random.choice(uniq, size=cnt, replace=False, p=probs))
        return sorted(selected)
    def _sample_bonus(self, group):
        pool = list(range(1,11)) if group=='low' else list(range(11,21))
        hist = self.df['bonus'].dropna().tail(100).values
        hist = [b for b in hist if b in pool]
        if hist:
            uniq, cnts = np.unique(hist, return_counts=True)
            return np.random.choice(uniq, p=cnts/cnts.sum())
        return np.random.choice(pool)

class TicketOptimizer:
    def __init__(self, main_pool, bonus_numbers, max_main, main_per_ticket):
        self.main_pool = [n for n in main_pool if n<=max_main] or list(range(1,max_main+1))
        self.bonus_pool = bonus_numbers if bonus_numbers else list(range(1,21))
        self.max_main = max_main
        self.main_per_ticket = main_per_ticket
    def generate_tickets(self, num_tickets, main_per_ticket=None, has_bonus=True):
        if main_per_ticket is None:
            main_per_ticket = self.main_per_ticket
        tickets = []
        for _ in range(num_tickets):
            if len(self.main_pool) >= main_per_ticket:
                main = np.random.choice(self.main_pool, size=main_per_ticket, replace=False)
            else:
                hot = self.main_pool.copy()
                remaining = main_per_ticket - len(hot)
                full = list(set(range(1,self.max_main+1)) - set(hot))
                extra = np.random.choice(full, size=remaining, replace=False)
                main = np.concatenate([hot, extra])
            ticket = {'main': sorted(main.tolist() if isinstance(main,np.ndarray) else main)}
            if has_bonus and self.bonus_pool:
                ticket['bonus'] = int(np.random.choice(self.bonus_pool))
            tickets.append(ticket)
        return tickets

class TicketChecker:
    PRIZE = {
        'daily': {5:'Jackpot',4:'2nd',3:'3rd',2:'4th'},
        'lotto': {6:'Jackpot',5:'2nd',4:'3rd',3:'4th',2:'5th'},
        'powerball': {5:'Jackpot',4:'2nd',3:'3rd',2:'4th',1:'5th'},
        'sportstake': {6:'Jackpot',5:'2nd',4:'3rd',3:'4th'},
        'sportstake_8': {8:'Jackpot',7:'2nd',6:'3rd',5:'4th'}
    }
    @staticmethod
    def check_ticket(ticket, draw, lottery_type):
        main_match = len(set(ticket['main']) & set(draw['main_numbers']))
        bonus_match = 0
        if ticket.get('bonus') is not None and draw.get('bonus') is not None:
            bonus_match = 1 if ticket['bonus'] == draw['bonus'] else 0
        tier_map = TicketChecker.PRIZE.get(lottery_type.split('_')[0], {})
        tier = tier_map.get(main_match, 'No win')
        return {'main_match': main_match, 'bonus_match': bonus_match, 'tier': tier}

# -------------------------------
# Main script
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lottery', default='all', choices=['all']+list(LOTTERY_CONFIG.keys()))
    args = parser.parse_args()

    # Locate data/raw relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'raw'
    if not data_dir.exists():
        print(f"ERROR: Data folder not found: {data_dir}")
        sys.exit(1)

    # Load all CSVs that exist
    available = [name for name in LOTTERY_CONFIG.keys() if (data_dir / f"{name}.csv").exists()]
    if not available:
        print("No CSV files found in data/raw/")
        print("Expected files: daily.csv, lotto.csv, etc.")
        sys.exit(1)

    for lottery in available:
        if args.lottery != 'all' and lottery != args.lottery:
            continue
        cfg = LOTTERY_CONFIG[lottery]
        has_bonus = cfg['has_bonus']
        max_main = cfg['max_main']
        main_count = cfg['main_count']

        # Load just this lottery
        df = pd.read_csv(data_dir / f"{lottery}.csv")
        # Parse main numbers
        if 'main_numbers' in df.columns:
            df['main_numbers'] = df['main_numbers'].apply(lambda x: [int(i) for i in str(x).strip('[]').split(',')])
        else:
            ball_cols = [c for c in df.columns if c.startswith(('ball','number','outcome')) and c[-1].isdigit()]
            ball_cols.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            df['main_numbers'] = df[ball_cols].values.tolist()
            df['main_numbers'] = df['main_numbers'].apply(lambda row: [int(x) for x in row])
        if 'bonus' not in df.columns:
            df['bonus'] = None
        else:
            df['bonus'] = df['bonus'].fillna(None)

        # Add group columns
        df = DataProcessor.add_group_columns(df, has_bonus, max_main)

        # Analyze and predict
        analyzer = GroupAnalyzer(df, has_bonus, use_last_n=None)  # all draws
        pred_groups = analyzer.predict_next_groups()
        predictor = NumberPredictor(df, has_bonus, max_main, main_count)
        pred_numbers = predictor.generate_prediction(pred_groups)

        # Get hot numbers
        all_mains = [n for sub in df['main_numbers'] for n in sub]
        top_mains = pd.Series(all_mains).value_counts().head(15).index.tolist()
        optimizer = TicketOptimizer(top_mains, pred_numbers.get('bonus'), max_main, main_count)

        print("\n" + "="*70)
        print(f"  {lottery.upper()}  |  Predicted group pattern: {pred_groups}")
        print("="*70)
        print(f"  🎯 Suggested focus numbers: {pred_numbers['main']}", end='')
        if has_bonus and pred_numbers.get('bonus'):
            print(f"  + Bonus: {pred_numbers['bonus']}")
        else:
            print()

        for num_tickets in [6, 13]:
            tickets = optimizer.generate_tickets(num_tickets, main_count, has_bonus)
            print(f"\n  ┌── {num_tickets} TICKET(S) ──")
            for i, t in enumerate(tickets):
                main_str = " ".join(f"{n:2d}" for n in t['main'])
                if has_bonus and t.get('bonus'):
                    print(f"  │ {i+1:2d}.  {main_str}   [B: {t['bonus']:2d}]")
                else:
                    print(f"  │ {i+1:2d}.  {main_str}")
            print("  └" + "─"*50)

            # Check against latest draw
            latest = df.iloc[-1].to_dict()
            winners = []
            for idx, tic in enumerate(tickets):
                res = TicketChecker.check_ticket(tic, latest, lottery)
                if res['tier'] != 'No win':
                    winners.append((idx+1, res))
            if winners:
                print(f"\n  ✨ Would have won against {latest.get('date','latest draw')}:")
                for idx, res in winners:
                    print(f"      Line {idx}: {res['tier']} (M:{res['main_match']}, B:{res['bonus_match']})")
            else:
                print(f"\n  (No win against {latest.get('date','latest draw')})")
        print("="*70)

if __name__ == "__main__":
    main()
