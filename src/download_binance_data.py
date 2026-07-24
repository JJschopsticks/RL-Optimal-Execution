import asyncio
import json
from datetime import datetime, UTC
from pathlib import Path
import websockets

# Binance WebSocket endpoints
DEPTH_STREAM = "wss://stream.binance.com:9443/ws/btcusdt@depth"
TRADE_STREAM = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# Data always lands in <project_root>/data, regardless of the CWD this script
# is launched from.
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Buffers to store updates before writing
depth_buffer = []
trade_buffer = []

# Rotation interval (in seconds)
ROTATE_INTERVAL = 60  # 1 minute


def get_timestamp():
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def get_date_folder():
    """Return today's folder name: YYYY-MM-DD"""
    return datetime.now(UTC).strftime("%Y-%m-%d")


def ensure_daily_folders():
    """Ensure the folder structure for today's date exists."""
    base = DATA_ROOT / get_date_folder()
    depth_path = base / "depth"
    trade_path = base / "trades"

    depth_path.mkdir(parents=True, exist_ok=True)
    trade_path.mkdir(parents=True, exist_ok=True)

    return depth_path, trade_path


async def collect_depth():
    async with websockets.connect(DEPTH_STREAM) as ws:
        print("Connected to depth stream")
        async for message in ws:
            data = json.loads(message)
            depth_buffer.append({
                "timestamp": get_timestamp(),
                "update": data
            })


async def collect_trades():
    async with websockets.connect(TRADE_STREAM) as ws:
        print("Connected to trade stream")
        async for message in ws:
            data = json.loads(message)
            trade_buffer.append({
                "timestamp": get_timestamp(),
                "trade": data
            })


async def rotate_files():
    while True:
        await asyncio.sleep(ROTATE_INTERVAL)

        depth_path, trade_path = ensure_daily_folders()

        # Depth rotation
        if depth_buffer:
            filename = depth_path / f"depth_{get_timestamp()}.json"
            with open(filename, "w") as f:
                json.dump(depth_buffer, f, indent=4)
            print(f"Saved depth file → {filename}")
            depth_buffer.clear()

        # Trade rotation
        if trade_buffer:
            filename = trade_path / f"trades_{get_timestamp()}.json"
            with open(filename, "w") as f:
                json.dump(trade_buffer, f, indent=4)
            print(f"Saved trades file → {filename}")
            trade_buffer.clear()


async def main():
    await asyncio.gather(
        collect_depth(),
        collect_trades(),
        rotate_files()
    )


if __name__ == "__main__":
    print("Starting Binance WebSocket collector...")
    asyncio.run(main())
