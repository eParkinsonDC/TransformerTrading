import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from models.timeseries_transformer import TimeSeriesTransformer


def train_transformer_model(
    model,
    train_loader,
    val_loader=None,
    lr=1e-3,
    epochs=20,
    device="cpu",
    save_best_path=None,  # <- new argument
):
    criterion = nn.MSELoss()
    # L2 Regularization with => weight_decay=1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Adding L1 Regularization for feature selection
    l1_lambda = 1e-4  # You can tune this value

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            # Add L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        mean_train_loss = np.mean(train_losses)

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    output_val = model(x_val)
                    loss_val = criterion(output_val, y_val)
                    val_losses.append(loss_val.item())
            mean_val_loss = np.mean(val_losses)
            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}"
            )

            # Save best model
            if save_best_path and mean_val_loss < best_val_loss:
                print(
                    f"[Checkpoint] Saving new best model: Val loss improved {best_val_loss:.6f} → {mean_val_loss:.6f}"
                )
                torch.save(model.state_dict(), save_best_path)
                best_val_loss = mean_val_loss
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {mean_train_loss:.6f}")
    return model


def get_underlying_dataset(data_loader):
    """
    Unwraps DataLoader.dataset recursively to get the underlying dataset.
    Handles both Subset and base Dataset.
    """
    dataset = data_loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def time_series_cross_validate(trader, k=5):
    """Performs expanding window cross-validation."""

    dataset = get_underlying_dataset(trader.train_loader)
    if not isinstance(dataset, torch.utils.data.Dataset):
        raise ValueError(
            "Expected train_loader.dataset to be a PyTorch Dataset, got: "
            f"{type(dataset)}"
        )
    total = len(dataset)
    fold_size = int(total / (k + 1))
    fold_metrics = []

    for fold in range(1, k + 1):
        train_end = fold * fold_size
        val_start = train_end
        val_end = val_start + fold_size

        train_subset = torch.utils.data.Subset(dataset, range(0, train_end))
        val_subset = torch.utils.data.Subset(
            dataset, range(val_start, min(val_end, total))
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=trader.batch_size, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=trader.batch_size, shuffle=False
        )

        model = TimeSeriesTransformer(
            feature_size=len(trader.feature_cols),
            num_layers=2,
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            seq_length=trader.seq_length,
            prediction_length=trader.pred_length,
        ).to(trader.device)

        model = train_transformer_model(
            model,
            train_loader,
            val_loader,
            lr=trader.lr,
            epochs=trader.epochs,
            device=trader.device,
        )
        # After training, compute val loss
        model.eval()
        val_losses = []
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(trader.device)
                y_val = y_val.to(trader.device)
                output_val = model(x_val)
                loss_val = criterion(output_val, y_val)
                val_losses.append(loss_val.item())
        mean_val_loss = np.mean(val_losses)
        print(f"CV Fold {fold}: Val Loss = {mean_val_loss:.6f}")
        fold_metrics.append(mean_val_loss)

    print(
        f"\nCross-Validation: Mean Val Loss = {np.mean(fold_metrics):.6f} (+/- {np.std(fold_metrics):.6f})"
    )
    return fold_metrics


def evaluate_model(
    model,
    test_loader,
    scaler,
    feature_cols,
    target_col_idx,
    window_width=10,
    start_index=0,
    pred_length=1,
    device="cpu",
):
    """
    Evaluates the model on test data and compares predictions with actual prices.
    Plots real vs. predicted values within a given window width and starting index.

    Parameters:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        scaler: MinMaxScaler (used to inverse transform predictions and real values).
        feature_cols: List of feature column names.
        target_col_idx: Index of the "Close" price in feature columns.
        window_width: Number of points to plot for real vs. predicted prices.
        start_index: The index in the test dataset from which to start plotting.
        pred_length: Number of future values predicted by the model.
        device: 'cpu' or 'cuda' for model inference.
    """

    model.eval()
    real_prices = []
    predicted_prices = []
    output_img_dir = "output_images"
    output_img = "evaluation_output.png"
    os.makedirs(output_img_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            # Get model predictions
            predictions = (
                model(x_batch).cpu().numpy()
            )  # shape: [batch_size, pred_length]
            y_batch = y_batch.cpu().numpy()  # shape: [batch_size, pred_length]

            for i in range(len(predictions)):
                # Create dummy inputs for inverse scaling
                dummy_pred = np.zeros((pred_length, len(feature_cols)))
                dummy_pred[:, target_col_idx] = predictions[
                    i
                ]  # Assign predicted future prices

                dummy_real = np.zeros((pred_length, len(feature_cols)))
                dummy_real[:, target_col_idx] = y_batch[i]  # Assign real future prices

                # Inverse transform both predicted and actual prices
                pred_inversed = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
                real_inversed = scaler.inverse_transform(dummy_real)[:, target_col_idx]

                # Store values
                predicted_prices.extend(pred_inversed)
                real_prices.extend(real_inversed)

    # Convert lists to numpy arrays
    real_prices = np.array(real_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()

    # -------------------------
    # Compute Accuracy Metrics
    # -------------------------
    mse = np.mean((real_prices - predicted_prices) ** 2)
    mae = np.mean(np.abs(real_prices - predicted_prices))

    print(f"Model Evaluation:\n  - Mean Squared Error (MSE): {mse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")

    # -------------------------
    # Adjust Start Index and Window Width for Plot
    # -------------------------
    if start_index < 0 or start_index >= len(real_prices):
        print(f"Warning: start_index {start_index} is out of bounds. Using 0 instead.")
        start_index = 0

    end_index = min(
        start_index + window_width * pred_length, len(real_prices)
    )  # Adjust for multi-step forecasts

    # -------------------------
    # Plot Real vs. Predicted Prices
    # -------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(start_index, end_index),
        real_prices[start_index:end_index],
        label="Real Close Prices",
        linestyle="dashed",
        marker="o",
    )
    plt.plot(
        range(start_index, end_index),
        predicted_prices[start_index:end_index],
        label="Predicted Close Prices",
        linestyle="-",
        marker="x",
    )
    plt.title(
        f"Real vs. Predicted Close Prices (From index {start_index}, {window_width} Windows, {pred_length} Steps Each)"
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Close Price")
    plt.legend()
    plt.savefig(os.path.join(output_img_dir, output_img))
    plt.close()


def plot_rolling_predictions(predictions, title="Rolling Forecast"):

    times = [pred["prediction_time"] for pred in predictions]
    last_closes = [pred["last_close"] for pred in predictions]
    pred_closes = [pred["next_close"] for pred in predictions]
    actions = [pred["action"].upper() for pred in predictions]
    output_img_dir = "output_images"
    output_img = "rolling_predictions.png"
    os.makedirs(output_img_dir, exist_ok=True)
    plt.figure(figsize=(16, 6))
    plt.plot(
        times, last_closes, label="Last Close", linestyle="--", color="blue", marker="o"
    )
    plt.plot(
        times,
        pred_closes,
        label="Predicted Next Close",
        linestyle="-",
        color="orange",
        marker="x",
    )

    # To deduplicate legend entries
    buy_shown, sell_shown, hold_shown = False, False, False

    for t, p, act in zip(times, pred_closes, actions):
        if act == "BUY" and not buy_shown:
            plt.scatter(t, p, color="green", s=80, label="BUY")
            buy_shown = True
        elif act == "BUY":
            plt.scatter(t, p, color="green", s=80)
        elif act == "SELL" and not sell_shown:
            plt.scatter(t, p, color="red", s=80, label="SELL")
            sell_shown = True
        elif act == "SELL":
            plt.scatter(t, p, color="red", s=80)
        elif act == "HOLD" and not hold_shown:
            plt.scatter(t, p, color="gray", s=40, label="HOLD")
            hold_shown = True
        else:
            plt.scatter(t, p, color="gray", s=40)

    plt.title(title)
    plt.xlabel("Prediction Time")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(output_img_dir, output_img))
    plt.close()
