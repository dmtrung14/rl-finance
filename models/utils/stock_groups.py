import yfinance
import pandas as pd
import numpy as np

class Tech:
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CSCO',
        'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM', 'TXN', 'IBM', 'AMAT', 'MU', 'NOW',
        'PYPL', 'ADI', 'LRCX', 'INTU', 'UBER', 'SHOP', 'SNPS', 'CDNS', 'WDAY', 'TEAM',
        'FTNT', 'PANW', 'ANET', 'DDOG', 'CRWD', 'ZS', 'ADSK', 'MRVL', 'DELL', 'HPQ',
        'VMW', 'CTSH', 'HPE', 'STX', 'WDC', 'KLAC', 'APH', 'GLW', 'KEYS', 'TEL'
    ]
