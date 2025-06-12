import base64
import csv
import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv


def fetch_candle_data(
    product_id, granularity, start, end, api_key, api_secret, api_passphrase
):
    method = "GET"
    request_path = f"/products/{product_id}/candles?granularity={granularity}&start={start}&end={end}"
    body = ""
    timestamp = str(int(time.time()))
    secret_decoded = base64.b64decode(api_secret)
    message = f"{timestamp}{method}{request_path}{body}".encode("utf-8")
    signature = hmac.new(secret_decoded, message, hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()

    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-PASSPHRASE": api_passphrase,
        "CB-ACCESS-SIGN": signature_b64,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }
    url = f"https://api.exchange.coinbase.com{request_path}"
    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Failed: {resp.status_code} {resp.text}")
        return []


def get_all_candles(
    product_id, granularity, start_dt, end_dt, api_key, api_secret, api_passphrase
):
    all_candles = []
    dt = start_dt
    fetch_count = 0
    print(f"Fetching candles for {product_id} from {start_dt} to {end_dt}...")
    while dt < end_dt:
        batch_end = min(dt + timedelta(seconds=granularity * 300), end_dt)
        start_iso = dt.isoformat()
        end_iso = batch_end.isoformat()
        print(
            f"  Downloading: {start_iso} to {end_iso} ({fetch_count} batches so far)",
            end="\r",
        )
        batch = fetch_candle_data(
            product_id,
            granularity,
            start_iso,
            end_iso,
            api_key,
            api_secret,
            api_passphrase,
        )
        if not batch:
            print("\nNo more data returned from API (may be at dataset start).")
            break
        all_candles += batch
        fetch_count += 1
        dt = batch_end
        time.sleep(0.15)
    print(f"\nTotal candles fetched: {len(all_candles)}")
    all_candles.sort(key=lambda x: x[0])
    return all_candles


def save_candles_to_csv(product_id, candles, start_dt=None, data_folder="data"):
    # Coinbase format: [ time, low, high, open, close, volume ]
    # Desired CSV: Gmt time,Open,High,Low,Close,Volume

    # Default: get the earliest timestamp in candles
    if start_dt is None and candles:
        first_ts = candles[0][0]
        start_dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
    elif start_dt is None:
        start_dt = datetime.now(tz=timezone.utc)
    start_str = start_dt.strftime("%Y%m%d")

    filename = f"{product_id}_1hour_candles_{start_str}.csv"

    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, filename)

    header = ["Gmt time", "Open", "High", "Low", "Close", "Volume"]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in candles:
            dt_str = datetime.fromtimestamp(row[0], tz=timezone.utc).strftime(
                "%d.%m.%Y %H:%M:%S.000"
            )
            writer.writerow([dt_str, row[3], row[2], row[1], row[4], row[5]])
    print(f"Saved to {file_path}")


def run_data_pipeline(arguments=None):
    if arguments is None:
        print(
            """No arguments (argsparse) have
            been passed to run_data_pipeline in cb_fetch_create_data.py"""
        )
        return False, arguments
    load_dotenv()
    api_key = os.getenv("CB_KEY")
    api_secret = os.getenv("CB_SECRET")
    api_passphrase = os.getenv("COINBASE_PASSPHRASE")
    product_id = arguments.product
    granularity = 3600  # 1 hour

    end_dt = datetime.now(timezone.utc)
    # supports floats (e.g., 1.5)
    start_dt = end_dt - timedelta(days=365 * arguments.years)

    candles = get_all_candles(
        product_id, granularity, start_dt, end_dt, api_key, api_secret, api_passphrase
    )
    if not candles:
        print("No candle data returned! Check your API credentials and product ID.")
        return False, arguments

    # --- Handle "years too large" case ---
    earliest_api_candle = datetime.fromtimestamp(candles[0][0], tz=timezone.utc)

    if earliest_api_candle > start_dt + timedelta(
        hours=1
    ):  # Give a little margin for exchange sync
        print(
            f"""
        \n[WARNING] Requested data starting from {start_dt:%Y-%m-%d}, but
        the earliest data from the API is {earliest_api_candle:%Y-%m-%d}.
        This is likely the full available dataset for this product."""
        )

    save_candles_to_csv(
        product_id,
        candles,
        start_dt=earliest_api_candle,
        data_folder=arguments.data_dir,
    )
    return True, arguments


if __name__ == "__main__":
    run_data_pipeline()
