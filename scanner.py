"""
Phase 2: Morning scanner (swing / delivery style, uses daily data).
Usage:  python scanner.py
Output: prints top picks and saves picks.json (our Android app will read this later)
"""
import json
from datetime import datetime

import pandas as pd
import yfinance as yf

SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "ASIANPAINT", "NTPC", "POWERGRID", "TATASTEEL",
    "M&M", "ULTRACEMCO", "WIPRO", "HCLTECH", "ONGC", "COALINDIA",
    "ADANIPORTS", "TECHM", "NESTLEIND", "TRENT",
]
TOP_N = 5


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def analyse(symbol, df):
    close = df["Close"]
    last = float(close.iloc[-1])
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    rsi14 = float(rsi(close).iloc[-1])
    vol_ratio = float(df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1])
    high20 = float(close.rolling(20).max().iloc[-1])
    atr14 = float(atr(df).iloc[-1])

    score, reasons = 0, []

    if last > sma50 > sma200:
        score += 2
        reasons.append("Strong uptrend (price > 50DMA > 200DMA)")
    elif last > sma50:
        score += 1
        reasons.append("Price above 50-day average")

    if 45 <= rsi14 <= 65:
        score += 1
        reasons.append(f"Healthy momentum (RSI {rsi14:.0f})")
    elif rsi14 > 75:
        score -= 1
        reasons.append(f"Overbought (RSI {rsi14:.0f}), risky to chase")

    if vol_ratio >= 1.2:
        score += 1
        reasons.append(f"Higher than usual volume ({vol_ratio:.1f}x)")

    if last >= 0.97 * high20:
        score += 1
        reasons.append("Near 20-day high (breakout zone)")

    # Risk management: stop-loss 1.5 x ATR below entry, target = 2 x risk
    stop = last - 1.5 * atr14
    target = last + 2 * (last - stop)

    return {
        "symbol": symbol,
        "score": score,
        "entry": round(last, 2),
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_pct": round((last - stop) / last * 100, 2),
        "rsi": round(rsi14, 1),
        "style": "Delivery / swing (days to weeks)",
        "reasons": reasons,
    }


def main():
    tickers = [s + ".NS" for s in SYMBOLS]
    print("Downloading data for", len(tickers), "stocks...")
    data = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)

    results = []
    for s in SYMBOLS:
        try:
            df = data[s + ".NS"].dropna()
            if len(df) < 210:
                print("Skipping", s, "(not enough data)")
                continue
            results.append(analyse(s, df))
        except Exception as e:
            print("Skipping", s, "->", e)

    results.sort(key=lambda r: r["score"], reverse=True)
    picks = results[:TOP_N]

    print("\n===== TODAY'S WATCHLIST (based on last close) =====")
    for r in picks:
        print(f"\n{r['symbol']}  | score {r['score']}/5")
        print(f"  Entry ~ Rs {r['entry']}   Stop-loss Rs {r['stop_loss']} "
              f"(-{r['risk_pct']}%)   Target Rs {r['target']}")
        for reason in r["reasons"]:
            print("   -", reason)

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "picks": picks}
    with open("picks.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved picks.json")
    print("Reminder: rule-based ideas, not guaranteed. Always use the stop-loss.\n")


if __name__ == "__main__":
    main()
