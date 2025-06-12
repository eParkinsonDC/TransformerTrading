
<div class="header" style="text-align: center; margin-top: 20px;">
    <h1>Transformer-Based Crypto/Forex Trading Signals</h1>
    <p>This project provides a complete pipeline for generating trading signals using a Transformer model on cryptocurrency/forex data.</p>
</div>

## A research-grade Python pipeline used for:

1. Fetching hourly candlestick data from Coinbase (or similar),

2. Building features and technical indicators,

3. Training a Transformer model to predict the next hour’s price,

4. Generating "buy" or "hold" trading signals for any specified period.

### Features
1. End-to-end data pipeline (fetch, preprocess, feature engineering, train, predict)

2. Model: Custom Transformer neural network (PyTorch)

3. Rolling-window prediction for today/tomorrow

4. Output trading signals to plain text for easy review

5. Modular structure for easy extension

### Project Structure
```plaintext
Transformer_TradingCB/
├── data/                           # Stores downloaded candlestick CSVs
├── models/
│   ├── timeseries_transformer.py   # Transformer model class
│   ├── forex_dataset.py            # PyTorch Dataset for time series
│   └── transformer_trading.py      # TradingTransformer pipeline class
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

## Quick Start
**1. Setup Python Environment**
- Make sure you have Python 3.11 installed (PyTorch does not yet support 3.13+ with CUDA as of October 2023).
- Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**2. Install Dependencies**

**Installation Steps**
1. Install requirements (CPU):

```bash
pip install -r requirements.txt
```

2. Or, for GPU (CUDA 12.1+):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
3. Install additional dependencies:

```bash
pip install pandas requests python-dotenv ta matplotlib
```

4. Install any missing dependencies as needed.


2. Set Coinbase API Keys
Create a .env file in your project directory with your API credentials (see mock.env for example):
The two essential env. variables are listed below:
```plaintext
CB_KEY= ...
CB_SECRET= ...
```


3. Fetch Data and Run Model
Run the full pipeline:

```batch
python -m scripts.main --years 6 --product ETH-EUR --data_dir data --new_data True
```

### Arguments

| Argument            | Type    | Description                                                                                  | Example                        |
|---------------------|---------|----------------------------------------------------------------------------------------------|--------------------------------|
| `--years`           | float   | How far back to fetch data (e.g., 1.5 for 18 months)                                         | `--years 2`                    |
| `--product`         | str     | Trading pair (e.g., BTC-USD, ETH-EUR)                                                        | `--product ETH-EUR`            |
| `--data_dir`        | str     | Folder to save CSVs and outputs                                                              | `--data_dir data`              |
| `--new_data`        | bool    | Whether to fetch new data (`True`) or use existing data in the directory (`False`)           | `--new_data True`              |
| `--batch_size`      | int     | Batch size for training and evaluation                                                       | `--batch_size 64`              |
| `--epochs`          | int     | Number of training epochs                                                                    | `--epochs 20`                  |
| `--lr`              | float   | Learning rate for optimizer                                                                  | `--lr 0.0005`                  |
| `--window_size`     | int     | Number of time steps in each input window                                                    | `--window_size 48`             |
| `--hidden_dim`      | int     | Hidden dimension size for the Transformer model                                              | `--hidden_dim 128`             |
| `--num_layers`      | int     | Number of Transformer encoder layers                                                         | `--num_layers 4`               |
| `--dropout`         | float   | Dropout rate for the Transformer model                                                       | `--dropout 0.2`                |
| `--device`          | str     | Device to use: `cpu` or `cuda`                                                               | `--device cuda`                |
| `--eval_only`       | bool    | Only run evaluation/prediction, skip training                                                | `--eval_only True`             |
| `--checkpoint`      | str     | Path to a model checkpoint to load for evaluation or resuming training                       | `--checkpoint outputs/model.pt` |
| `--seed`            | int     | Random seed for reproducibility                                                              | `--seed 42`                    |
| `--log_interval`    | int     | How often (in steps) to log training progress                                                | `--log_interval 10`            |
| `--output_signals`  | bool    | Whether to output trading signals to file                                                    | `--output_signals True`        |
| `--cv_folds`       | int     | Number of cross-validation folds for evaluation (default: 1, no CV)                          | `--cv_folds 5`                 |
| `--rolling_window`  | int     | Size of the rolling window for per-hour predictions (default: 24)                             | `--rolling_window 24`          |
You can combine these arguments as needed when running the pipeline.

### Outputs
1. **signals.txt** : Latest "buy"/"hold"/"sell" signals for each product/period in your dataset.

2. **windowed_signals_[product]_[date].txt**: Per-hour rolling predictions (useful for simulation, backtesting, or live monitoring).


### Key Modules

| Module                                   | Description                                                      |
|-------------------------------------------|------------------------------------------------------------------|
| `models/forex_dataset.py`                 | Converts feature-engineered CSV into a sequence dataset.         |
| `scripts/cb_fetch_and_create_data.py`     | Authenticates and fetches hourly candlestick data from Coinbase. |
| `models/transformer_trading.py`           | Main pipeline: loads data, trains, predicts, outputs signals.    |
| `models/timeseries_transformer.py`        | Custom Transformer model for time series prediction.             |
| `utils/technical_indicators.py`          | Contains functions to compute technical indicators.              |


### Extending or Adapting
Add new technical indicators:
Edit add_technical_indicators() in transformer_trading.py.

Change trading logic:
Edit predict_next() in TradingTransformer.

Use another data source:
Replace/fork the fetch script.

### Dependencies
See requirements.txt (typical stack: torch, numpy, pandas, ta, requests, python-dotenv, matplotlib).

### Disclaimer
This codebase is for research and education only. Not financial advice.
Always evaluate performance on unseen data and understand model/market risks before using in any trading application.

### Author
Eduardo Parkinson de Castro

### Questions, feature requests, or issues?
Open an issue or ask away!

