import numpy as np

class TicketOptimizer:
    def __init__(self, main_pool, bonus_numbers=None, max_main=52, main_per_ticket=6):
        """
        main_pool: list of recommended main numbers (e.g. top 15 hot numbers)
        bonus_numbers: list of recommended bonus numbers (if any)
        max_main: maximum allowed main number
        main_per_ticket: how many main numbers per ticket (5 or 6)
        """
        self.main_pool = [n for n in main_pool if n <= max_main]
        if not self.main_pool:
            self.main_pool = list(range(1, max_main+1))
        self.bonus_pool = bonus_numbers if bonus_numbers else list(range(1, 21))
        self.max_main = max_main
        self.main_per_ticket = main_per_ticket
    
    def generate_tickets(self, num_tickets=1, main_per_ticket=None, has_bonus=True):
        """Generate num_tickets tickets, each a dict with 'main' and optionally 'bonus'."""
        if main_per_ticket is None:
            main_per_ticket = self.main_per_ticket
        
        tickets = []
        for _ in range(num_tickets):
            # Sample main numbers without replacement from the pool
            # If pool is smaller than needed, fallback to full range
            if len(self.main_pool) >= main_per_ticket:
                main = np.random.choice(self.main_pool, size=main_per_ticket, replace=False)
            else:
                # Not enough hot numbers, fill remaining from full range
                hot = self.main_pool.copy()
                remaining = main_per_ticket - len(hot)
                full_range = list(set(range(1, self.max_main+1)) - set(hot))
                extra = np.random.choice(full_range, size=remaining, replace=False)
                main = np.concatenate([hot, extra])
            main = sorted(main)
            
            ticket = {'main': main.tolist() if isinstance(main, np.ndarray) else main}
            if has_bonus and self.bonus_pool:
                bonus = np.random.choice(self.bonus_pool)
                ticket['bonus'] = int(bonus)
            tickets.append(ticket)
        return tickets
