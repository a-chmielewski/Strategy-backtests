import csv
import os
import sys
from collections import OrderedDict
from datetime import datetime

# Directory where per-strategy result files are stored
RESULTS_DIR = "results"
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
    # Basic identifier sanity
    if not (strategy and coinpair and timeframe) or leverage is None:
        print(f"[log_result ERROR] Missing identifier: "
              f"{strategy=}, {coinpair=}, {timeframe=}, {leverage=}",
              file=sys.stderr)
        return

    # Build our new row
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

    # Skip if all metric fields are empty
    metric_fields = [
        "start", "end", "duration", "equity_final", "return_pct",
        "num_trades", "win_rate", "avg_trade", "best_trade", "worst_trade",
        "max_drawdown", "avg_drawdown", "sharpe_ratio", "profit_factor"
    ]
    if not any(new_row[f] not in (None, "") for f in metric_fields):
        print("[log_result] No metrics present, skip writing.")
        return

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, f"{strategy}.csv")

    # Load existing rows, if any
    rows_map = OrderedDict()
    if os.path.isfile(file_path):
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == FIELDNAMES:
                for ex in reader:
                    # require identifiers
                    if not all(ex.get(k) for k in ("strategy","coinpair","timeframe","leverage")):
                        continue
                    if not is_valid_date(ex.get("start","")):
                        continue
                    key = (
                        ex["strategy"], ex["coinpair"],
                        ex["timeframe"], ex["leverage"]
                    )
                    rows_map[key] = ex
            else:
                print(f"[log_result WARNING] Header mismatch in {file_path}, starting fresh.")

    # Insert or replace our new row
    key = (new_row["strategy"], new_row["coinpair"],
           new_row["timeframe"], new_row["leverage"])
    action = "Replacing" if key in rows_map else "Appending"
    print(f"[log_result] {action} row for key={key}")
    rows_map[key] = new_row

    # Write back out
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows_map.values():
            # ensure only the known fields are output
            clean = {k: row.get(k, "") for k in FIELDNAMES}
            writer.writerow(clean)
