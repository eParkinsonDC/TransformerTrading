import os
import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe. Handles short data by filling with NaN.
    """
    import ta
    import numpy as np

    # RSI, ADX, ATR need at least 14 rows
    if len(df) >= 14:
        df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
        df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
        df["atr"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], window=14
        )
    else:
        df["rsi"] = np.nan
        df["adx"] = np.nan
        df["atr"] = np.nan

    # Bollinger Bands, MA, slope need at least 20 rows
    if len(df) >= 20:
        bollinger = ta.volatility.BollingerBands(
            close=df["Close"], window=20, window_dev=2
        )
        df["bb_high"] = bollinger.bollinger_hband()
        df["bb_low"] = bollinger.bollinger_lband()
        df["ma_20"] = df["Close"].rolling(window=20).mean()
        df["ma_20_slope"] = df["ma_20"].diff()
    else:
        df["bb_high"] = np.nan
        df["bb_low"] = np.nan
        df["ma_20"] = np.nan
        df["ma_20_slope"] = np.nan

    # EMA 12 needs at least 12 rows, EMA 26 needs at least 26
    df["ema_12"] = (
        ta.trend.ema_indicator(df["Close"], window=12) if len(df) >= 12 else np.nan
    )
    df["ema_26"] = (
        ta.trend.ema_indicator(df["Close"], window=26) if len(df) >= 26 else np.nan
    )

    # MACD (will handle missing values internally)
    df["macd"] = ta.trend.macd(df["Close"])
    df["macd_signal"] = ta.trend.macd_signal(df["Close"])

    # Stochastic Oscillator
    df["stoch_k"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
    df["stoch_d"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"])

    # On-Balance Volume
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # 1h, 24h returns
    df["returns_1h"] = df["Close"].pct_change(periods=1)
    df["returns_24h"] = df["Close"].pct_change(periods=24)

    # 24h rolling volatility
    df["volatility_24h"] = df["returns_1h"].rolling(window=24).std()

    # Fill missing values just in case
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df


def select_important_features(df, target_col="Close", n_features=None):

    always_keep = [
        "Open",
        "High",
        "Low",
        "Close",
        "rsi",
        "bb_high",
        "bb_low",
        "ma_20",
        "ma_20_slope",
    ]
    # Remove any that aren't in dataframe columns
    always_keep = [col for col in always_keep if col in df.columns]

    # Candidate features: everything except time/target
    all_possible = [col for col in df.columns if col not in [target_col, "GMT_TIME"]]
    # Remove always_keep from candidate list
    candidate = [col for col in all_possible if col not in always_keep]

    X = df[always_keep + candidate].fillna(0)
    y = df[target_col]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({"feature": X.columns, "importance": importances})

    # Remove always_keep from ranking (they’re added anyway)
    ranked = importance_df[~importance_df["feature"].isin(always_keep)]
    ranked = ranked.sort_values("importance", ascending=False)

    # Pick up to (n_features - len(always_keep)) most important
    extra_needed = max(0, n_features - len(always_keep))
    top_learned = ranked["feature"].tolist()[:extra_needed]

    # Combine, keeping order, no duplicates
    final_features = always_keep + [f for f in top_learned if f not in always_keep]

    # If fewer features than n_features, just use what you have
    return final_features[:n_features]


def select_and_scale_features(df, feature_cols=None, n_features=None):
    if feature_cols is None:
        feature_cols = select_important_features(
            df, target_col="Close", n_features=n_features
        )
    # Always keep "Close" (Target feature)
    if "Close" not in feature_cols:
        feature_cols.append("Close")
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler, feature_cols


def save_features_and_config(
    feature_cols, config, output_dir="outputs", prefix="model"
):
    os.makedirs(output_dir, exist_ok=True)
    config_out = {
        "config": config,
    }
    
    if 'feature_cols' not in config:
        config_out['feature_cols'] = feature_cols
        config_path = os.path.join(output_dir, f"{prefix}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, default=str)
    print(f"Saved features and config to {config_path}")
