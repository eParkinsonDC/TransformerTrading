<div align="center" style="margin-top: 20px;">
  <h1>Transformer-Based Crypto/Forex Trading Signals</h1>
  <p>
    This project provides a complete pipeline for generating trading signals using a Transformer model on cryptocurrency/forex data.
  </p>
</div>

## A research-grade Python pipeline used for:

1. Fetching hourly candlestick data from Coinbase (or similar)
2. Building features and technical indicators
3. Training a Transformer model to predict the next hour’s price
4. Generating "buy", "hold", or "sell" trading signals for any specified period

---

### Features

1. End-to-end data pipeline (fetch, preprocess, feature engineering, train, predict)
2. Model: Custom Transformer neural network (PyTorch)
3. Rolling-window prediction for today/tomorrow
4. Output trading signals to plain text for easy review
5. Modular structure for easy extension

---

### Project Structure

```plaintext
Transformer_TradingCB/
├── data/                           # Stores downloaded candlestick CSVs
├── models/
│   ├── timeseries_transformer.py   # Transformer model class
│   ├── forex_dataset.py            # PyTorch Dataset for time series
│   └── trading_transformer.py      # TradingTransformer pipeline class
├── scripts/
│   ├── main.py                     # Command-line entry point
│   ├── cb_fetch_and_create_data.py # Data fetching & preprocessing
│   └── train_and_eval.py           # Training & evaluation utilities
├── outputs/                        # Stores output signals and predictions
├── utils/                          # Utility functions (e.g., technical indicators)
├── requirements.txt                # Python package requirements
├── .env.example                    # Example environment variables
└── README.md                       # (This file)
```

---

## Quick Start

### 1. Setup Python Environment

* **Python 3.11 recommended** (PyTorch CUDA as of 2024 may not support 3.13+).
* Use a virtual environment:

```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

---

### 2. Install Dependencies

**CPU:**

```bash
pip install -r requirements.txt
```

**For CUDA (GPU):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Make sure you also have:**

```bash
pip install pandas requests python-dotenv ta matplotlib
```

---

### 3. Set Coinbase API Keys

* Copy `.env.example` to `.env` and add your API credentials:

```plaintext
CB_KEY=...
CB_SECRET=...
```

---

### 4. Fetch Data and Run Model

Example (fetch + train + predict):

```bash
python -m scripts.main --years 6 --product ETH-EUR --data_dir data --new_data
```

---

## Arguments

| Argument           | Type  | Description                                                              | Example                  |
| ------------------ | ----- | ------------------------------------------------------------------------ | ------------------------ |
| `--years`          | float | How far back to fetch data (years, can be decimal, e.g. 1.5 = 18 months) | `--years 2`              |
| `--product`        | str   | Trading pair, e.g. `ETH-EUR`, `BTC-USD`                                  | `--product ETH-EUR`      |
| `--data_dir`       | str   | Folder to save CSVs and outputs                                          | `--data_dir data`        |
| `--new_data`       | flag  | Fetch new data from API                                                  | `--new_data`             |
| `--batch_size`     | int   | Batch size for training and evaluation                                   | `--batch_size 64`        |
| `--epochs`         | int   | Number of training epochs                                                | `--epochs 20`            |
| `--learning_rate`  | float | Learning rate for optimizer                                              | `--learning_rate 0.0005` |
| `--seq_length`     | int   | Number of timesteps in each input window                                 | `--seq_length 48`        |
| `--d_model`        | int   | Hidden dimension size for the Transformer model                          | `--d_model 128`          |
| `--num_layers`     | int   | Number of Transformer encoder layers                                     | `--num_layers 4`         |
| `--dropout`        | float | Dropout rate for the Transformer model                                   | `--dropout 0.2`          |
| `--device`         | str   | Device to use: `cpu` or `cuda`                                           | `--device cuda`          |
| `--cv_folds`       | int   | Number of cross-validation folds for evaluation (default: 1 = no CV)     | `--cv_folds 5`           |
| `--rolling_window` | int   | Size of the rolling window for per-hour predictions (default: 24)        | `--rolling_window 24`    |
| ...                |       | (see code for full set)                                                  |                          |

---

## Outputs

1. **signals.txt:** Latest "buy"/"hold"/"sell" signals for each product/period.
2. **windowed\_signals\_\[product]\_\[date].txt:** Per-hour rolling predictions (for simulation/backtesting/live monitoring).
3. **outputs/\[...].json:** Model configs and selected features for reproducibility.

---

## Key Modules

| Module                                | Description                                               |
| ------------------------------------- | --------------------------------------------------------- |
| `models/forex_dataset.py`             | Converts engineered CSVs to PyTorch sequence datasets.    |
| `scripts/cb_fetch_and_create_data.py` | Authenticates & fetches hourly data from Coinbase.        |
| `models/trading_transformer.py`       | Main pipeline: data load, train, predict, output signals. |
| `models/timeseries_transformer.py`    | Transformer model for time series regression.             |
| `utils/model_utils.py`                | Technical indicators, feature selection, and helpers.     |

---

## Extending or Adapting

* **Add technical indicators:** Edit `add_technical_indicators()` in `utils/model_utils.py`.
* **Change trading logic:** Edit `predict_next()` in `TradingTransformer`.
* **New data source:** Replace or adapt the data fetch script.

---

## Dependencies

See `requirements.txt`. Typical stack:
`torch`, `numpy`, `pandas`, `ta`, `requests`, `python-dotenv`, `matplotlib`, etc.

---

## Disclaimer

**For research and education only. Not financial advice.**
Always test on unseen data and understand all model/market risks before any real-world use.

---

## Author

Eduardo Parkinson de Castro

---
