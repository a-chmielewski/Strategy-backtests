import csv
import os
import sys
from collections import OrderedDict
from datetime import datetime

RESULTS_FILE = "all_backtest_results.csv"
FIELDNAMES = [
    "strategy", "coinpair", "timeframe", "leverage",
    "start", "end", "duration", "equity_final", "return_pct",
    "num_trades", "win_rate", "avg_trade", "best_trade", "worst_trade",
    "max_drawdown", "avg_drawdown", "sharpe_ratio", "profit_factor"
]

def is_valid_date(s: str) -> bool:
    """Return True if s is YYYY-MM-DD."""
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except Exception:
        return False

def log_result(strategy, coinpair, timeframe, leverage, results):
    # DEBUG AT ENTRY
    print(f"[log_result] strategy={strategy!r}, coinpair={coinpair!r}, "
          f"timeframe={timeframe!r}, leverage={leverage!r}, "
          f"start={results.get('Start')!r}")

    # 1) Basic identifier sanity
    if not strategy or not coinpair or not timeframe or leverage is None:
        print(f"[log_result ERROR] Missing identifier: "
              f"{strategy=}, {coinpair=}, {timeframe=}, {leverage=}",
              file=sys.stderr)
        return

    # 2) Build our new row
    new_row = {
        "strategy":      str(strategy),
        "coinpair":      str(coinpair),
        "timeframe":     str(timeframe),
        "leverage":      str(leverage),
        "start":         results.get("Start", ""),
        "end":           results.get("End", ""),
        "duration":      results.get("Duration", ""),
        "equity_final":  results.get("Equity Final [$]", ""),
        "return_pct":    results.get("Return [%]", ""),
        "num_trades":    results.get("# Trades", ""),
        "win_rate":      results.get("Win Rate [%]", ""),
        "avg_trade":     results.get("Avg. Trade", ""),
        "best_trade":    results.get("Best Trade", ""),
        "worst_trade":   results.get("Worst Trade", ""),
        "max_drawdown":  results.get("Max. Drawdown [%]", ""),
        "avg_drawdown":  results.get("Avg. Drawdown [%]", ""),
        "sharpe_ratio":  results.get("Sharpe Ratio", ""),
        "profit_factor": results.get("Profit Factor", ""),
    }

    # 3) Skip entirely if metrics are all empty
    value_fields = [
        "start", "end", "duration", "equity_final", "return_pct",
        "num_trades", "win_rate", "avg_trade", "best_trade", "worst_trade",
        "max_drawdown", "avg_drawdown", "sharpe_ratio", "profit_factor"
    ]
    if not any(new_row[f] not in (None, "") for f in value_fields):
        print("[log_result] No metrics present, skip writing.")
        return

    # 4) Load existing rows into an OrderedDict keyed by our 4 IDs,
    #    but drop any row whose 'start' isn't a valid date.
    rows_map = OrderedDict()
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, newline="") as f:
            reader = csv.DictReader(f)
            # If headers are corrupt, warn & skip loading
            if reader.fieldnames != FIELDNAMES:
                print(f"[log_result WARNING] Header mismatch: {reader.fieldnames!r}")
            else:
                for ex in reader:
                    # must have all 4 keys
                    if not all(ex.get(k) for k in ("strategy","coinpair","timeframe","leverage")):
                        continue
                    # filter out any row with non-ISO start
                    if not is_valid_date(ex.get("start","")):
                        continue

                    key = (ex["strategy"], ex["coinpair"], ex["timeframe"], ex["leverage"])
                    rows_map[key] = ex

    # 5) Insert or replace our new row
    key = (new_row["strategy"], new_row["coinpair"], new_row["timeframe"], new_row["leverage"])
    action = "Replacing" if key in rows_map else "Appending"
    print(f"[log_result] {action} row for key={key}")
    rows_map[key] = new_row

    # 6) Write everything back out
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        # Clean each row dict so it only has the FIELDNAMES keys:
        cleaned_rows = [
            { key: row.get(key, "") for key in FIELDNAMES }
            for row in rows_map.values()
        ]

        writer.writerows(cleaned_rows)