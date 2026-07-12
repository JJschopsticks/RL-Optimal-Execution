# order_book_reconstruction.py

import os
import json
from decimal import Decimal
from typing import Dict, Tuple, Generator

PROCESSED_FILE = "processed_files.txt"
CLEANED_DIR = "cleaned"


class OrderBook:
    def __init__(self):
        self.bids: Dict[Decimal, Decimal] = {}
        self.asks: Dict[Decimal, Decimal] = {}

    def apply_update(self, update: dict):
        # Apply bid updates
        for price_str, qty_str in update.get("b", []):
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        # Apply ask updates
        for price_str, qty_str in update.get("a", []):
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def best_bid(self) -> Tuple[Decimal, Decimal] | None:
        return max(self.bids.items(), key=lambda x: x[0], default=None)

    def best_ask(self) -> Tuple[Decimal, Decimal] | None:
        return min(self.asks.items(), key=lambda x: x[0], default=None)

    def midprice(self) -> Decimal | None:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb and ba:
            return (bb[0] + ba[0]) / 2
        return None


def load_processed_files() -> set:
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())


def save_processed_file(path: str):
    with open(PROCESSED_FILE, "a") as f:
        f.write(path + "\n")


def stream_new_depth_files(folder_path: str) -> Generator[Tuple[str, str, dict], None, None]:
    processed = load_processed_files()
    files = sorted(os.listdir(folder_path))

    for filename in files:
        if not filename.endswith(".json"):
            continue

        full_path = os.path.join(folder_path, filename)

        if full_path in processed:
            continue

        with open(full_path, "r") as f:
            events = json.load(f)
            for event in events:
                yield full_path, event["timestamp"], event["update"]

        save_processed_file(full_path)


def find_all_depth_folders(base_path="data"):
    depth_folders = []
    for date_folder in os.listdir(base_path):
        full_date_path = os.path.join(base_path, date_folder)
        if not os.path.isdir(full_date_path):
            continue

        depth_path = os.path.join(full_date_path, "depth")
        if os.path.isdir(depth_path):
            depth_folders.append((date_folder, depth_path))

    return sorted(depth_folders)


def main():
    os.makedirs(CLEANED_DIR, exist_ok=True)

    depth_folders = find_all_depth_folders("data")

    for date_str, depth_folder in depth_folders:
        cleaned_path = os.path.join(CLEANED_DIR, f"{date_str}.jsonl")

        print(f"Processing folder: {depth_folder}")
        print(f"Writing cleaned output to: {cleaned_path}")

        book = OrderBook()

        with open(cleaned_path, "a") as out:
            for full_path, timestamp, update in stream_new_depth_files(depth_folder):
                book.apply_update(update)

                bb = book.best_bid()
                ba = book.best_ask()
                mid = book.midprice()

                out.write(json.dumps({
                    "timestamp": timestamp,
                    "best_bid": [str(bb[0]), str(bb[1])] if bb else None,
                    "best_ask": [str(ba[0]), str(ba[1])] if ba else None,
                    "midprice": str(mid) if mid else None
                }) + "\n")


if __name__ == "__main__":
    main()
