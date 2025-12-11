import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

ATR_PERIOD = 10
MULTIPLIER = 3.0
VOL_TARGET = 0.70  

SYMBOL = "BTC-USD"
LOG_FILE = "trade_log.csv"

import json
try:
    with open('params.json', 'r') as f:
        params = json.load(f)
        ATR_PERIOD = params.get('period', 10)
        MULTIPLIER = params.get('multiplier', 3.0)
        VOL_TARGET = params.get('vol_target', 0.70)
        print(f"Loaded Params: {ATR_PERIOD}/{MULTIPLIER}, VolTarget={VOL_TARGET}")
except Exception as e:
    print(f"Params error: {e}. Using Defaults.")

def fetch_latest_data(symbol=SYMBOL):
    print(f"Fetching latest data for {symbol}...")

    df = yf.download(symbol, period="6mo", progress=False) 
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_supertrend(df_input, period=ATR_PERIOD, multiplier=MULTIPLIER):
    df = df_input.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()

    hl2 = (df['High'] + df['Low']) / 2
    df['BasicUpper'] = hl2 + (multiplier * df['ATR'])
    df['BasicLower'] = hl2 - (multiplier * df['ATR'])

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    trend = np.zeros(n)
    
    close = df['Close'].values
    basic_upper = df['BasicUpper'].values
    basic_lower = df['BasicLower'].values
      
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    
    for i in range(1, n):
        if np.isnan(basic_upper[i]):
            final_upper[i] = np.nan
        elif basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        if np.isnan(basic_lower[i]):
            final_lower[i] = np.nan
        elif basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        prev = trend[i-1]

        if np.isnan(final_upper[i]) or np.isnan(final_lower[i]):
            trend[i] = 0
        elif prev == -1 and close[i] > final_upper[i]:
            trend[i] = 1
        elif prev == 1 and close[i] < final_lower[i]:
            trend[i] = -1
        else:
            trend[i] = prev
            
        if trend[i] == 0 and not np.isnan(final_upper[i]):
            trend[i] = 1
            
    df['PctChange'] = df['Close'].pct_change()
    df['AnnVol'] = df['PctChange'].rolling(20).std() * np.sqrt(365)

    df['SuperTrend'] = trend
    df['FinalUpper'] = final_upper
    df['FinalLower'] = final_lower
    return df

def run_forward_test():
    df = fetch_latest_data()
    df = calculate_supertrend(df)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    date_str = latest.name.strftime('%Y-%m-%d')
    price = latest['Close']
    trend_now = latest['SuperTrend']
    trend_prev = prev['SuperTrend']
    
    action = "HOLD"
    
    if trend_prev == -1 and trend_now == 1:
        action = "BUY"
    elif trend_prev == 1 and trend_now == -1:
        action = "SELL"
    else:
        if trend_now == 1:
            action = "HOLD LONG"
        else:
            action = "HOLD CASH"
            
    current_vol = latest['AnnVol']
    if np.isnan(current_vol) or current_vol == 0:
        current_vol = 1.0 
    
    raw_alloc = VOL_TARGET / current_vol
    allocation_pct = min(raw_alloc, 1.0)
    
    print(f"--- Forward Test: {date_str} ---")
    print(f"Price: ${price:,.2f}")
    print(f"Trend: {'Bullish (1)' if trend_now==1 else 'Bearish (-1)'}")
    print(f"Volatility: {current_vol*100:.1f}%")
    print(f"Rec. Allocation: {allocation_pct*100:.1f}%")
    print(f"Action: {action}")

    log_entry = {
        'Date': date_str,
        'Price': price,
        'Trend': trend_now,
        'Action': action,
        'Volatility': round(current_vol, 4),
        'Allocation': round(allocation_pct, 4),
        'ATR_Params': f"{ATR_PERIOD}/{MULTIPLIER}",
        'RunTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    row = pd.DataFrame([log_entry])
    
    if not os.path.exists(LOG_FILE):
        row.to_csv(LOG_FILE, index=False)
        print(f"Created new log file: {LOG_FILE}")
    else:
        existing = pd.read_csv(LOG_FILE)
        if date_str in existing['Date'].astype(str).values:
            print("Log for this date already exists. Skipping append.")
        else:
            row.to_csv(LOG_FILE, mode='a', header=False, index=False)
            print(f"Logged to {LOG_FILE}")
            
            print(f"Logged to {LOG_FILE}")
            
    update_readme(price, date_str)

def update_readme(current_price, current_date_str):
    if not os.path.exists(LOG_FILE):
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return

    capital = 100.0
    btc_amount = 0.0
    in_position = False
    first_buy_date = None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    for index, row in df.iterrows():
        action = row['Action']
        price = float(row['Price'])
        
        if action == "BUY" and not in_position:
            btc_amount = capital / price
            capital = 0.0
            in_position = True
            if first_buy_date is None:
                first_buy_date = row['Date']
        
        elif action == "SELL" and in_position:
            capital = btc_amount * price
            btc_amount = 0.0
            in_position = False

    if in_position:
        current_val = btc_amount * float(current_price)
    else:
        current_val = capital

    total_profit_pct = ((current_val - 100.0) / 100.0) * 100.0

    monthly_row = ""
    warning_text = "Monthly average profit starts calculation 30 days after the first purchase."
    
    if first_buy_date is not None:
        current_dt = pd.to_datetime(current_date_str)
        days_diff = (current_dt - first_buy_date).days
        
        if days_diff > 30:
            months = days_diff / 30.0
            if months > 0:
                monthly_avg = total_profit_pct / months
                monthly_row = f"| Monthly Average Profit | {monthly_avg:.2f}% |\n"
            warning_text = "" 
            warning_text = "Monthly metric active (>30 days)."

    markdown_content = f"""# Forward Test Results 📊

| Metric | Value |
| :--- | :--- |
| **Total Profit** | **{total_profit_pct:.2f}%** |
{monthly_row}

> ⚠️ *Note: Monthly average profit only appears 30 days after the first executed order.*
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print("Updated README.md with Forward Stats")

if __name__ == "__main__":
    run_forward_test()
