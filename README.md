# Smart Order Router (RL-Based Optimal Execution)
A reinforcement-learning system that learns how to liquidate a large crypto position against a live, reconstructed limit order book while minimizing market impact and slippage trained with PPO, validated against real historical and live Binance data, and served through a live paper-trading dashboard.
This project simulates the "optimal execution" problem that trading desks solve every day: if you need to sell a large amount of an asset, dumping it all at once crashes the price against you. A smarter agent breaks the order into pieces and paces them based on real-time order book signals. Here, that pacing policy is learned rather than hand-coded, and it's benchmarked head-to-head against classic execution baselines (TWAP, dump-all, no-trade) on the same data.

Sample Live Session:
<img width="2322" height="1076" alt="SORLiveSession" src="https://github.com/user-attachments/assets/567b0b36-4f15-4f58-abe6-8d16afad64c8" />
<img width="2278" height="666" alt="SORLossGraph" src="https://github.com/user-attachments/assets/99925cc4-c7f3-4a2c-a073-a68474885664" />

# Key features
- Custom Gym environment (sor_env.py, sor_live_env.py) exposing a 37-dimensional feature vector per tick: order book state, order flow imbalance, microprice, depth imbalance, TWAP/VWAP tracking, realized volatility, and execution cost/impact metrics.
- PPO training pipeline (train_rl.py) built on Stable-Baselines3, with VecNormalize observation/reward scaling, checkpointing, and evaluation callbacks against a genuinely unseen date.
- Live order book reconstruction from raw Binance depth-diff events, including update-ID gap detection and crossed-book handling.
- FastAPI backend (src/api/server.py) that runs and streams a live paper-trading session over WebSocket, enforcing the exact (quantity, horizon) ranges the deployed model was validated on.
- React + TypeScript dashboard (webapp/) for watching a live session unfold: KPIs, connection status, session history, and live charts.
- Policy evaluation harness (eval_policies.py) for running and comparing any policy against the same replay engine.

# Tech stack
- RL / ML: Python, Stable-Baselines3 (PPO), Gymnasium, NumPy
- Backend: FastAPI, WebSockets, Pydantic
- Data: Binance WebSocket API, custom L2 order book reconstruction
- Frontend: React, TypeScript, Vite
- Precision: Python Decimal used throughout the execution core to avoid floating-point error in price/quantity math

# How to start
0. Install dependencies
bash
pip install -r requirements.txt

1. Collect and clean order book data
bash
cd src
python download_binance_data.py      # streams live Binance depth/trade data to data/
python clean_l2.py 2026-07-11        # reconstructs an L2 book for a given date

(Note: download_binance_data.py connects to Binance's global WebSocket endpoints, which aren't accessible from every region Binance.US does not expose equivalent full-depth order book streams, so this step requires network access to Binance's main international API.)

2. Train the PPO agent
bash
python train_rl.py --train-dates 2026-07-11 2026-07-12 --held-out-date 2026-07-21

3. Evaluate against baselines
bash
python eval_policies.py

4. Run the live paper-trading dashboard
bash
Backend (from src/)
python -m uvicorn api.server:app --reload

Frontend (from webapp/)
npm install
npm run dev

# Disclaimer
This is a research/educational project. It only ever places simulated orders against replayed or live-streamed market data — there is no exchange API key or real order-placement code anywhere in this repo.

# License
MIT
