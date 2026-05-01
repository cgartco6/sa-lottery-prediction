class TicketChecker:
    PRIZE_TIERS = {
        'daily': {5: 'Jackpot', 4: '2nd', 3: '3rd', 2: '4th'},
        'lotto': {6: 'Jackpot', 5: '2nd', 4: '3rd', 3: '4th', 2: '5th'},
        'lotto_plus1': {6: 'Jackpot', 5: '2nd', 4: '3rd', 3: '4th', 2: '5th'},
        'lotto_plus2': {6: 'Jackpot', 5: '2nd', 4: '3rd', 3: '4th', 2: '5th'},
        'powerball': {5: 'Jackpot', 4: '2nd', 3: '3rd', 2: '4th', 1: '5th'},
        'powerball_plus': {5: 'Jackpot', 4: '2nd', 3: '3rd', 2: '4th', 1: '5th'},
        'daily_lotto': {5: 'Jackpot', 4: '2nd', 3: '3rd', 2: '4th'},
        'daily_lotto_plus1': {5: 'Jackpot', 4: '2nd', 3: '3rd', 2: '4th'},
        'sportstake': {6: 'Jackpot', 5: '2nd', 4: '3rd', 3: '4th'},  # adjust based on real rules
        'sportstake_8': {8: 'Jackpot', 7: '2nd', 6: '3rd', 5: '4th'}
    }
    
    @staticmethod
    def check_ticket(ticket, draw, lottery_type):
        """
        ticket: dict with 'main' list and optional 'bonus'
        draw: dict with 'main_numbers' list and optional 'bonus'
        Returns dict with main_match, bonus_match, tier.
        """
        main_match = len(set(ticket['main']) & set(draw['main_numbers']))
        bonus_match = 0
        if ticket.get('bonus') is not None and draw.get('bonus') is not None:
            bonus_match = 1 if ticket['bonus'] == draw['bonus'] else 0
        
        # Simple tier mapping - you can refine per lottery
        tier_map = TicketChecker.PRIZE_TIERS.get(lottery_type, {})
        if lottery_type in ['lotto','lotto_plus1','lotto_plus2','powerball','powerball_plus']:
            # For lotto with bonus, we might need combined key. Simplified for now.
            if main_match >= 3 and bonus_match:
                key = main_match  # treat bonus as extra; you can adjust
            else:
                key = main_match
        else:
            key = main_match
        
        tier = tier_map.get(key, 'No win')
        return {'main_match': main_match, 'bonus_match': bonus_match, 'tier': tier}
