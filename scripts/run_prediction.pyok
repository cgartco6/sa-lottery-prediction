#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import sys
import re

# ============================================================
# OFFICIAL FILENAMES (exactly as downloaded)
# ============================================================
LOTTERY_RULES = {
    "DAILYLOTTO-RESULTS":          {"main_count": 5, "max_main": 36, "has_bonus": False},
    "DAILYLOTTO-PLUS-RESULTS":     {"main_count": 5, "max_main": 36, "has_bonus": False},
    "LOTTO-RESULTS":               {"main_count": 6, "max_main": 52, "has_bonus": True, "bonus_max": 20},
    "LOTTO-PLUS1-RESULTS":         {"main_count": 6, "max_main": 52, "has_bonus": True, "bonus_max": 20},
    "LOTTO-PLUS2-RESULTS":         {"main_count": 6, "max_main": 52, "has_bonus": True, "bonus_max": 20},
    "POWERBALL-RESULTS":           {"main_count": 5, "max_main": 50, "has_bonus": True, "bonus_max": 20},
    "POWERBALL-PLUS-RESULTS":      {"main_count": 5, "max_main": 50, "has_bonus": True, "bonus_max": 20},
    "SPORTSTAKE-1X2-RESULTS":      {"main_count": 6, "max_main":  4, "has_bonus": False},
    "SPORTSTAKE-SS08-RESULTS":     {"main_count": 8, "max_main":  4, "has_bonus": False},
}
# ============================================================

def find_number_columns(df, main_count, filename):
    """
    Find ball columns: expects 'ball1', 'ball2', ... up to main_count.
    """
    # First, look for exact 'ballX' pattern
    candidates = []
    for i in range(1, main_count + 1):
        col_name = f"ball{i}"
        if col_name in df.columns:
            candidates.append(col_name)
        else:
            # Also try alternative: 'ball0i' for single digit? Not needed.
            pass
    
    if len(candidates) == main_count:
        return candidates
    
    # Fallback: any column with 'ball' followed by digits, sorted by the number
    all_ball_cols = []
    for col in df.columns:
        match = re.search(r'ball(\d+)', col, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            all_ball_cols.append((num, col))
    all_ball_cols.sort()
    if len(all_ball_cols) >= main_count:
        return [col for _, col in all_ball_cols[:main_count]]
    
    # Last resort: take first N numeric columns that are not dates/payouts
    exclude = re.compile(r'date|payout|div|draw|next|jackpot', re.IGNORECASE)
    numeric_cols = []
    for col in df.columns:
        if exclude.search(col):
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        except:
            pass
    if len(numeric_cols) >= main_count:
        return numeric_cols[:main_count]
    
    raise ValueError(f"Cannot find {main_count} ball columns in {filename}")

def find_bonus_column(df, filename):
    """Find bonus ball column (usually 'bonusball' or 'bonus ball')."""
    for col in df.columns:
        if 'bonus' in col.lower():
            # Ensure it's not a payout column
            if not any(x in col.lower() for x in ['payout', 'div']):
                return col
    return None

# ------------------------------------------------------------------
# Helper classes (same logic, but clean)
# ------------------------------------------------------------------
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
        if 'SPORTSTAKE-1X2' in lottery_type:
            base = 'sportstake'
        elif 'SPORTSTAKE-SS08' in lottery_type:
            base = 'sportstake_8'
        elif 'DAILYLOTTO' in lottery_type:
            base = 'daily'
        elif 'LOTTO' in lottery_type:
            base = 'lotto'
        elif 'POWERBALL' in lottery_type:
            base = 'powerball'
        else:
            base = 'lotto'
        tier_map = TicketChecker.PRIZE.get(base, {})
        tier = tier_map.get(main_match, 'No win')
        return {'main_match': main_match, 'bonus_match': bonus_match, 'tier': tier}

# ------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lottery', default='all', help='Specific lottery (e.g., LOTTO-RESULTS)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'raw'
    if not data_dir.exists():
        print(f"ERROR: Data folder not found: {data_dir}")
        sys.exit(1)

    for csv_name, cfg in LOTTERY_RULES.items():
        if args.lottery != 'all' and csv_name != args.lottery:
            continue
        csv_path = data_dir / f"{csv_name}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        has_bonus = cfg['has_bonus']
        max_main = cfg['max_main']
        main_count = cfg['main_count']

        df = pd.read_csv(csv_path)

        # Find main number columns
        try:
            main_cols = find_number_columns(df, main_count, csv_name)
        except Exception as e:
            print(f"Error processing {csv_name}: {e}")
            print("Columns found:", df.columns.tolist())
            continue

        # Convert to numeric, coerce errors to NaN
        for col in main_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract main numbers row by row, drop rows with any NaN in main_cols
        df['main_numbers'] = df[main_cols].values.tolist()
        df['main_numbers'] = df['main_numbers'].apply(lambda row: [int(x) for x in row if pd.notna(x)])
        # Keep only rows where we have exactly main_count numbers
        df = df[df['main_numbers'].apply(len) == main_count]
        if len(df) == 0:
            print(f"No valid rows (all have missing numbers) in {csv_name}")
            continue

        # Bonus column
        if has_bonus:
            bonus_col = find_bonus_column(df, csv_name)
            if bonus_col:
                df['bonus'] = pd.to_numeric(df[bonus_col], errors='coerce')
            else:
                df['bonus'] = None
        else:
            df['bonus'] = None

        # Add group columns
        df = DataProcessor.add_group_columns(df, has_bonus, max_main)

        # Analyze and predict
        analyzer = GroupAnalyzer(df, has_bonus, use_last_n=None)
        pred_groups = analyzer.predict_next_groups()
        predictor = NumberPredictor(df, has_bonus, max_main, main_count)
        pred_numbers = predictor.generate_prediction(pred_groups)

        all_mains = [n for sub in df['main_numbers'] for n in sub]
        top_mains = pd.Series(all_mains).value_counts().head(15).index.tolist()
        optimizer = TicketOptimizer(top_mains, pred_numbers.get('bonus'), max_main, main_count)

        print("="*70)
        print(f"  {csv_name}  |  Predicted group pattern: {pred_groups}")
        print("="*70)
        focus = [int(x) for x in pred_numbers['main']]
        print(f"  🎯 Suggested focus numbers: {focus}", end='')
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
                res = TicketChecker.check_ticket(tic, latest, csv_name)
                if res['tier'] != 'No win':
                    winners.append((idx+1, res))
            if winners:
                # Try to find a date column
                date_col = None
                for col in df.columns:
                    if 'date' in col.lower():
                        date_col = col
                        break
                draw_date = latest.get(date_col, 'latest draw') if date_col else 'latest draw'
                print(f"\n  ✨ Would have won against {draw_date}:")
                for idx, res in winners:
                    print(f"      Line {idx}: {res['tier']} (M:{res['main_match']}, B:{res['bonus_match']})")
            else:
                print(f"\n  (No win against latest draw)")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
