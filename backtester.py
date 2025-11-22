import numpy as np


class SimpleBacktester:
    def run_strategy(self, prices, labels, long_clusters, short_clusters):
        cash = 10000
        position = 0
        equity_curve = []
        trades = 0
        
        for i in range(len(prices)):
            current_price = prices[i]
            current_label = labels[i]
            
            if current_label in long_clusters:
                if position == 0:
                    position = cash / current_price
                    cash = 0
                    trades += 1
            
            elif current_label in short_clusters:
                if position > 0:
                    cash = position * current_price
                    position = 0
                    trades += 1
            
            portfolio_value = cash + (position * current_price)
            equity_curve.append(portfolio_value)
        
        return np.array(equity_curve), trades

