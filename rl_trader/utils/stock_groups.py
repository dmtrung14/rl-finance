import yfinance
import pandas as pd
import numpy as np

class Tickers:
    tech = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'TSM',
        'ASML', 'TCEHY'
        ]

    finance = [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'PNC', 'USB', 'AXP', 'COF', 'SPGI'
    ]

    energy = [
        'XOM', 'CVX', 'RDS-A', 'PTR', 'TOT', 'BP', 'ENB', 'COP', 'EQNR', 'PBR', 'SLB', 'EOG'
    ]