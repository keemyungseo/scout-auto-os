"""
Scout Season2 - Historical Data Collector (parameterized by date)

Research only. Reuses top10 gainer learning pipeline for any study date.
"""

import argparse
import csv
import sys
import time
import urllib.error
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10

KST = timezone(timedelta(hours=9))
SCAN_HOURS = (9, 11, 13, 15, 17, 19, 21, 23)
LOGS_DIR = Path("logs")


def scan_times_for_date(date_str: str) -> list[tuple[str, str, datetime]]:
    rows = []
    for hour in SCAN_HOURS:
        kst_str = f"{date_str} {hour:02d}:00:00"
        utc_dt = datetime.strptime(kst_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST).astimezone(timezone.utc)
        rows.append((kst_str, t10.format_time_utc(utc_dt), utc_dt))
    return rows


def collect_date(date_str: str, top_n: int = 10) -> Path:
    output = LOGS_DIR / f"top10_gainer_learning_{date_str.replace('-', '')}.csv"
    if output.exists():
        print(f"[skip] {date_str} already exists: {output}")
        return output

    scan_times = scan_times_for_date(date_str)
    eligible = t10.get_eligible_symbols()
    if not eligible:
        raise RuntimeError("no eligible symbols")

    print(f"Collecting {date_str} TOP{top_n} | universe={len(eligible)}")
    all_records: list[t10.Top10Record] = []
    symbols = sorted(eligible)

    for scan_kst, scan_utc, scan_dt in scan_times:
        print(f"  scan {scan_kst}")
        end_ms = int(scan_dt.timestamp() * 1000)
        candidates: list[tuple[str, dict]] = []

        for index, symbol in enumerate(symbols, start=1):
            try:
                klines = t10.fetch_klines_before(symbol, t10.INTERVAL_2H, end_ms, t10.RANKING_KLINES_2H)
                ranking = t10.compute_24h_ranking(klines)
                if ranking is not None:
                    candidates.append((symbol, ranking))
            except urllib.error.HTTPError:
                continue
            time.sleep(t10.API_SLEEP_SEC)

        candidates.sort(key=lambda item: item[1]["return_24h_percent"], reverse=True)
        for rank, (symbol, ranking) in enumerate(candidates[:top_n], start=1):
            try:
                klines_2h = t10.fetch_klines_before(symbol, t10.INTERVAL_2H, end_ms, t10.ANALYSIS_KLINES_2H)
                klines_1h = t10.fetch_klines_before(symbol, t10.INTERVAL_1H, end_ms, t10.ANALYSIS_KLINES_1H)
                forward = t10.measure_forward(symbol, scan_dt, ranking["price_at_scan"])
                record = t10.build_record(
                    scan_kst, scan_utc, scan_dt, rank, symbol, ranking, klines_2h, klines_1h, forward
                )
                if record is not None:
                    all_records.append(record)
            except urllib.error.HTTPError:
                continue
            time.sleep(t10.API_SLEEP_SEC)

    if not all_records:
        print(f"[warn] no records for {date_str}")
        return output

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(t10.Top10Record) if field.name != "scan_dt"]
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(t10.record_to_row(record) for record in all_records)
    print(f"Saved {len(all_records)} rows -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    collect_date(args.date, args.top_n)


if __name__ == "__main__":
    main()
