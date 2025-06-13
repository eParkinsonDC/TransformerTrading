import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from models.forex_dataset import ForexDataset
from models.timeseries_transformer import TimeSeriesTransformer
from scripts.train_and_eval import (
    evaluate_model,
    plot_rolling_predictions,
    walk_forward_time_series_cv_gridsearch,
    train_transformer_model,
)
from utils.model_utils import (
    select_important_features,
    select_and_scale_features,
    add_technical_indicators,
    save_features_and_config,
)


class TradingTransformer:
    def __init__(
        self,
        csv_file,
        seq_length=30,
        pred_length=1,
        batch_size=32,
        epochs=20,
        lr=1e-3,
        device=None,
        cv_folds=2,
        n_features=8,
    ):
        self.csv_file = csv_file
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[INFO] Using device: {self.device}")
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.target_col_idx = None
        self.model_config = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.n_features = n_features
        self.cv_folds = cv_folds

    def create_config(self):
        self.model_config = {
            "model_type": self.__class__.__name__,
            "cuda_device": self.device,
            "seq_length": self.seq_length,
            "pred_length": self.pred_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "feature_cols": self.feature_cols,
            "target_col_idx": self.target_col_idx,
            "n_features": self.n_features,
            "cv_folds": self.cv_folds,
            "model": self.model.__class__.__name__ if self.model else None,
            "device": self.device,
            "csv_file": self.csv_file,
            "scaler": self.scaler.__class__.__name__ if self.scaler else None,
        }
        return self.model_config

    def load_and_prepare(self):
        df = pd.read_csv(self.csv_file)
        df["GMT_TIME"] = pd.to_datetime(df["GMT_TIME"], format="%d.%m.%Y %H:%M:%S.%f")
        df.sort_values(by="GMT_TIME", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = add_technical_indicators(df)

        dataset_length = len(df)
        train_size = int(dataset_length * 0.8)
        val_size = int(dataset_length * 0.1)
        test_size = dataset_length - train_size - val_size

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size :]

        # Core feature set you always want to keep
        must_have = [
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
        feature_cols = select_important_features(
            train_df, target_col="Close", n_features=self.n_features
        )
        for f in must_have:
            if f not in feature_cols:
                feature_cols.append(f)
        if "Close" not in feature_cols:
            feature_cols.append("Close")
        scaler = MinMaxScaler().fit(train_df[feature_cols])
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.target_col_idx = self.feature_cols.index("Close")
        self.create_config()
        prefix = f"{os.path.splitext(os.path.basename(self.csv_file))[0]}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        save_features_and_config(
            self.feature_cols, self.model_config, output_dir="outputs", prefix=prefix
        )

        train_scaled = scaler.transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])
        test_scaled = scaler.transform(test_df[feature_cols])

        train_dataset = ForexDataset(
            train_scaled,
            self.seq_length,
            self.pred_length,
            len(self.feature_cols),
            self.target_col_idx,
        )
        val_dataset = ForexDataset(
            val_scaled,
            self.seq_length,
            self.pred_length,
            len(self.feature_cols),
            self.target_col_idx,
        )
        test_dataset = ForexDataset(
            test_scaled,
            self.seq_length,
            self.pred_length,
            len(self.feature_cols),
            self.target_col_idx,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        print(f"Loaded data. Feature columns: {self.feature_cols}")

    def build_model(self):
        self.model = TimeSeriesTransformer(
            feature_size=len(self.feature_cols),
            num_layers=2,
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            seq_length=self.seq_length,
            prediction_length=self.pred_length,
        ).to(self.device)

    def train(self):
        if self.model is None:
            self.build_model()
        self.model = train_transformer_model(
            self.model,
            self.train_loader,
            self.val_loader,
            lr=self.lr,
            epochs=self.epochs,
            device=self.device,
        )

    def evaluate(self, window_width=45, start_index=70):
        evaluate_model(
            self.model,
            self.test_loader,
            self.scaler,
            self.feature_cols,
            self.target_col_idx,
            window_width=window_width,
            start_index=start_index,
            pred_length=self.pred_length,
            device=self.device,
        )

    def predict_next(self):
        df = pd.read_csv(self.csv_file)
        df["GMT_TIME"] = pd.to_datetime(df["GMT_TIME"], format="%d.%m.%Y %H:%M:%S.%f")
        df = df.sort_values("GMT_TIME").reset_index(drop=True)
        df = add_technical_indicators(df)
        data_scaled, _, _ = select_and_scale_features(df, self.feature_cols)
        x_input = data_scaled[-self.seq_length :]
        x_tensor = (
            torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(x_tensor).cpu().numpy().flatten()
        dummy = np.zeros((self.pred_length, len(self.feature_cols)))
        dummy[:, self.target_col_idx] = pred_scaled
        next_close = self.scaler.inverse_transform(dummy)[:, self.target_col_idx][0]
        last_close = df["Close"].iloc[-1]
        if next_close > last_close:
            action = "BUY"
        elif next_close < last_close:
            action = "SELL"
        else:
            action = "HOLD"
        print(
            f"Last close: {last_close:.5f} | Predicted next close: {next_close:.5f} => Action: {action}"
        )
        return {
            "next_close": float(next_close),
            "last_close": float(last_close),
            "action": action,
        }

    def recursive_predict_future_window(self, steps=24, verbose=True):
        df = pd.read_csv(self.csv_file)
        df["GMT_TIME"] = pd.to_datetime(df["GMT_TIME"], format="%d.%m.%Y %H:%M:%S.%f")
        df = df.sort_values("GMT_TIME").reset_index(drop=True)
        df_pred = df.copy()
        results = []
        for i in range(steps):
            window_df = df_pred.iloc[-self.seq_length :].copy()
            window_df = add_technical_indicators(window_df)
            data_scaled, _, _ = select_and_scale_features(window_df, self.feature_cols)
            x_tensor = (
                torch.tensor(data_scaled, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(x_tensor).cpu().numpy().flatten()
            dummy = np.zeros((self.pred_length, len(self.feature_cols)))
            dummy[:, self.target_col_idx] = pred_scaled
            next_close = self.scaler.inverse_transform(dummy)[:, self.target_col_idx][0]
            last_row = window_df.iloc[-1].copy()
            new_time = last_row["GMT_TIME"] + pd.Timedelta(hours=1)
            new_row = last_row.copy()
            new_row["GMT_TIME"] = new_time
            new_row["Open"] = last_row["Close"]
            new_row["High"] = max(last_row["Close"], next_close)
            new_row["Low"] = min(last_row["Close"], next_close)
            new_row["Close"] = next_close
            new_row["Volume"] = last_row["Volume"]
            df_pred = pd.concat([df_pred, pd.DataFrame([new_row])], ignore_index=True)
            if next_close > last_row["Close"]:
                action = "BUY"
            elif next_close < last_row["Close"]:
                action = "SELL"
            else:
                action = "HOLD"
            if verbose:
                print(
                    f"{new_time:%d.%m.%Y %H:00:00.000}: Last close {last_row['Close']:.5f} | Predicted next close {next_close:.5f} => {action}"
                )
            results.append(
                {
                    "prediction_time": new_time,
                    "last_close": float(last_row["Close"]),
                    "next_close": float(next_close),
                    "action": action,
                }
            )
        return results

    @staticmethod
    def batch_run(data_dir="data", output_file="signals.txt", **kwargs):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not any(fname.endswith(".csv") for fname in os.listdir(data_dir)):
            print(
                f"Warning: The data directory '{data_dir}' is empty or contains no CSV files."
            )
            return

        # Only keys expected by TradingTransformer __init__
        MODEL_ARGS = {
            "seq_length",
            "pred_length",
            "batch_size",
            "epochs",
            "lr",
            "device",
            "cv_folds",
            "n_features",
        }

        # Allow both 'learning_rate' and 'lr' CLI arguments
        if "learning_rate" in kwargs:
            kwargs["lr"] = kwargs.pop("learning_rate")
        model_args = {k: v for k, v in kwargs.items() if k in MODEL_ARGS}

        for csv_file in os.listdir(data_dir):
            if not csv_file.endswith(".csv"):
                continue
            csv_file_path = os.path.join(data_dir, csv_file)
            print(f"\nProcessing file: {csv_file_path}")
            trader = TradingTransformer(csv_file_path, **model_args)
            trader.load_and_prepare()

            # ---- Walk-forward CV using current CSV ----
            df = pd.read_csv(csv_file_path)
            # Accept both time column styles
            time_col = "Gmt time" if "Gmt time" in df.columns else "GMT_TIME"
            df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M:%S.%f")
            df = df.sort_values(time_col).reset_index(drop=True)

            must_have = [
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

            param_grid = {
                "num_layers": [2, 4, 6],
                "d_model": [64, 128, 256],
                "nhead": [8, 16, 32],
                "learning_rate": [1e-3, 5e-4],
                "dropout": [0.1, 0.2],
                "dim_feedforward": [256, 512],
                "n_features": [
                    kwargs.get("n_features", 30),
                    kwargs.get("n_features", 30) - 5,
                    kwargs.get("n_features", 30) - 8,
                ],
            }

            cv_losses = walk_forward_time_series_cv_gridsearch(
                df=df,
                param_grid=param_grid,
                n_folds=kwargs.get("cv_folds", 5),
                model_class=TimeSeriesTransformer,
                verbose=True,
            )
            print(
                f"Cross-validation mean val loss: {np.mean(cv_losses):.6f} +/- {np.std(cv_losses):.6f}"
            )

            trader.build_model()
            best_model_path = os.path.join(
                output_dir, f"best_model_{os.path.splitext(csv_file)[0]}.pt"
            )
            trader.model = train_transformer_model(
                trader.model,
                trader.train_loader,
                trader.val_loader,
                lr=trader.lr,
                epochs=trader.epochs,
                device=trader.device,
                save_best_path=best_model_path,
            )
            trader.model.load_state_dict(
                torch.load(best_model_path, map_location=trader.device)
            )
            trader.model.eval()
            trader.evaluate()
            signal = trader.predict_next()
            df_eval = pd.read_csv(csv_file_path, encoding="utf-8")
            last_time = df_eval[time_col].iloc[-1]
            if signal["action"].upper() == "BUY":
                line = f"BUY at {last_time}: Predicted next close {signal['next_close']:.5f} > Last close {signal['last_close']:.5f}\n"
            elif signal["action"].upper() == "SELL":
                line = f"SELL at {last_time}: Predicted next close {signal['next_close']:.5f} < Last close {signal['last_close']:.5f}\n"
            else:
                line = f"HOLD at {last_time}: Predicted next close {signal['next_close']:.5f} == Last close {signal['last_close']:.5f}\n"
            print(line.strip())
            signal_file = os.path.join(output_dir, output_file)
            with open(signal_file, "a", encoding="utf-8") as f_out:
                f_out.write(line)
            print(f"\nSummary signals written to {output_file}")
            window_predictions = trader.recursive_predict_future_window(
                steps=kwargs.get("rolling_window", 24), verbose=True
            )
            plot_rolling_predictions(window_predictions)
            windowed_filename = f"windowed_signals_{os.path.splitext(csv_file)[0]}.txt"
            window_file_path = os.path.join(output_dir, windowed_filename)
            with open(window_file_path, "a", encoding="utf-8") as f_out2:
                for pred in window_predictions:
                    line2 = (
                        f"{pred['prediction_time']:%Y-%m-%d %H:%M}: "
                        f"Last close {pred['last_close']:.5f} | "
                        f"Predicted next close {pred['next_close']:.5f} => "
                        f"{pred['action'].upper()}\n"
                    )
                    f_out2.write(line2)
            print(f"\nWindowed signals written to {windowed_filename}")


if __name__ == "__main__":
    TradingTransformer.batch_run(data_dir="data", output_file="signals.txt")
