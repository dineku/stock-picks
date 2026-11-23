"""
Scanner v3 (swing / delivery style, uses daily data).
- Reads the stock list from symbols.txt (one NSE symbol per line).
- Saves picks.json with: top picks + latest indicators ("signals") for every stock.
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))
TOP_N = 10
BATCH = 40

# Used only if symbols.txt is missing
DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK",
]


def load_symbols():
    try:
        with open("symbols.txt") as f:
            syms = [line.strip().upper() for line in f
                    if line.strip() and not line.strip().startswith("#")]
        if syms:
            return list(dict.fromkeys(syms))  # remove duplicates, keep order
    except FileNotFoundError:
        pass
    print("symbols.txt not found, using the default list.")
    return DEFAULT_SYMBOLS


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
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
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

    pick = {
        "symbol": symbol,
        "score": score,
        "entry": round(last, 2),
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_pct": round((last - stop) / last * 100, 2),
        "rsi": round(rsi14, 1),
        "vol_ratio": round(vol_ratio, 2),
        "style": "Delivery / swing (days to weeks)",
        "reasons": reasons,
    }
    signal = {
        "close": round(last, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "rsi": round(rsi14, 1),
        "atr": round(atr14, 2),
        "date": str(df.index[-1].date()),
    }
    return pick, signal


def main():
    symbols = load_symbols()
    print("Scanning", len(symbols), "stocks...")

    results, signals = [], {}
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        tickers = [s + ".NS" for s in batch]
        try:
            data = yf.download(tickers, period="1y", interval="1d",
                               group_by="ticker", auto_adjust=True,
                               progress=False, threads=True)
        except Exception as e:
            print("Batch failed:", e)
            continue

        for s in batch:
            try:
                df = data[s + ".NS"].dropna()
                if len(df) < 210:
                    print("Skipping", s, "(not enough data or wrong symbol)")
                    continue
                pick, signal = analyse(s, df)
                results.append(pick)
                signals[s] = signal
            except Exception as e:
                print("Skipping", s, "->", e)
        time.sleep(1)  # be gentle with Yahoo

    if not results:
        # Fail loudly so the old picks.json stays in place
        print("ERROR: no data could be analysed. Keeping the old picks.json.")
        sys.exit(1)

    results.sort(key=lambda r: (r["score"], r["vol_ratio"]), reverse=True)
    picks = results[:TOP_N]

    print("\n===== TODAY'S WATCHLIST (based on last close) =====")
    for r in picks:
        print(f"\n{r['symbol']}  | score {r['score']}/5")
        print(f"  Entry ~ Rs {r['entry']}   Stop-loss Rs {r['stop_loss']} "
              f"(-{r['risk_pct']}%)   Target Rs {r['target']}")
        for reason in r["reasons"]:
            print("   -", reason)

    out = {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "as_of": max(sig["date"] for sig in signals.values()),
        "picks": picks,
        "signals": signals,
    }
    with open("picks.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved picks.json with", len(signals), "stocks")
    print("Reminder: rule-based ideas, not guaranteed. Always use the stop-loss.\n")


if __name__ == "__main__":
    main()
