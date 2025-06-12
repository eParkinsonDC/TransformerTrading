import argparse
from models.trading_transformer import TradingTransformer
from scripts.cb_fetch_and_create_data import run_data_pipeline


def get_default_args():
    return {
        "years": 8,
        "product": "ETH-EUR",
        "data_dir": "data",
        "new_data": False,
        "n_features": 8,
        "model_dir": "outputs",
        "cv_folds": 2,
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "dropout": 0.1,
        "seq_length": 30,
        "prediction_length": 1,
        "num_layers": 2,
        "d_model": 64,
        "nhead": 8,
        "dim_feedforward": 256,
        "rolling_window": 60,
    }


def get_non_default_args(args, defaults):
    user_args = vars(args)
    # Map 'learning_rate' -> 'lr' if needed
    if "learning_rate" in user_args:
        user_args["lr"] = user_args["learning_rate"]
        del user_args["learning_rate"]
    return {k: v for k, v in user_args.items() if k in defaults and v != defaults[k]}


def main():
    parser = argparse.ArgumentParser(
        description="Download Coinbase hourly candle data as CSV."
    )
    # Data Arguments
    parser.add_argument("--years", type=float, default=8)
    parser.add_argument("--product", type=str, default="ETH-EUR")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--new_data", action="store_true")
    parser.add_argument("--model_dir", type=str, default="outputs")
    # Training and evaluation arguments
    parser.add_argument("--n_features", type=int, default=8)
    parser.add_argument("--cv_folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rolling_window", type=int, default=60)
    # Model Arguments
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=256)

    args = parser.parse_args()
    defaults = get_default_args()
    non_default_args = get_non_default_args(args, defaults)
    print("Non-default arguments:", non_default_args)  # Debugging; remove if desired

    if args.new_data:
        finished, _ = run_data_pipeline(args)
        if finished:
            TradingTransformer.batch_run(args.data_dir, **non_default_args)
        else:
            raise RuntimeError("Error running data Pipeline")
    else:
        TradingTransformer.batch_run(args.data_dir, **non_default_args)


if __name__ == "__main__":
    main()
