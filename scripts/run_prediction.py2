#!/usr/bin/env python3
import argparse
import yaml
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.group_analyzer import GroupAnalyzer
from src.predictor import NumberPredictor
from src.ticket_optimizer import TicketOptimizer
from src.ticket_checker import TicketChecker

def main():
    parser = argparse.ArgumentParser(description='SA Lottery Predictor')
    parser.add_argument('--lottery', type=str, default='all',
                        choices=['all','daily','lotto','lotto_plus1','lotto_plus2',
                                 'powerball','powerball_plus','daily_lotto',
                                 'daily_lotto_plus1','sportstake','sportstake_8'],
                        help='Specific lottery to run')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    data_dict = DataLoader.load_all(config_lotteries=config['lotteries'])
    if not data_dict:
        print("No data loaded. Check CSV files in data/raw/")
        return

    # Standard ticket counts: 6 lines and 13 lines
    standard_ticket_counts = [6, 13]

    for lottery, df in data_dict.items():
        if args.lottery != 'all' and lottery != args.lottery:
            continue

        info = config['lotteries'][lottery]
        has_bonus = info['has_bonus']
        max_main = info['max_main']
        main_count = info['main_count']
        use_last_n = config['analysis'].get('use_last_n_draws')

        # Add group columns
        df = DataProcessor.add_group_columns(df, has_bonus, max_main)

        # Analyse group patterns
        analyzer = GroupAnalyzer(df, has_bonus, use_last_n)
        predicted_groups = analyzer.predict_next_groups()

        # Predict numbers
        predictor = NumberPredictor(df, has_bonus, max_main, main_count)
        prediction = predictor.generate_prediction(predicted_groups)

        # Prepare hot number pool for ticket generation
        all_mains = [n for sub in df['main_numbers'] for n in sub]
        top_mains = pd.Series(all_mains).value_counts().head(15).index.tolist()
        optimizer = TicketOptimizer(top_mains, prediction.get('bonus'), max_main, main_count)

        # Display header
        print("\n" + "="*70)
        print(f"  {lottery.upper()}  |  Predicted pattern: {predicted_groups}")
        print("="*70)
        print(f"  🎯 Suggested focus numbers: {prediction['main']}", end='')
        if has_bonus and prediction.get('bonus'):
            print(f"  + Bonus: {prediction['bonus']}")
        else:
            print()

        # Generate and display ticket sets
        for num_tickets in standard_ticket_counts:
            tickets = optimizer.generate_tickets(num_tickets=num_tickets,
                                                 main_per_ticket=main_count,
                                                 has_bonus=has_bonus)
            print(f"\n  ┌── {num_tickets} TICKET(S) ──")
            for i, t in enumerate(tickets):
                main_str = " ".join(f"{n:2d}" for n in t['main'])
                if has_bonus and t.get('bonus'):
                    print(f"  │ {i+1:2d}.  {main_str}   [B: {t['bonus']:2d}]")
                else:
                    print(f"  │ {i+1:2d}.  {main_str}")
            print("  └" + "─"*50)

            # Optional: check against latest draw
            latest_draw = df.iloc[-1].to_dict()
            winners = []
            for idx, ticket in enumerate(tickets):
                result = TicketChecker.check_ticket(ticket, latest_draw, lottery)
                if result['tier'] != 'No win':
                    winners.append((idx+1, result))
            if winners:
                print(f"\n  ✨ Would have won against {latest_draw.get('date', 'latest draw')}:")
                for idx, res in winners:
                    print(f"      Line {idx}: {res['tier']} (M:{res['main_match']}, B:{res['bonus_match']})")
            else:
                print(f"\n  (No win against {latest_draw.get('date', 'latest draw')})")

        print("="*70)

if __name__ == "__main__":
    main()
