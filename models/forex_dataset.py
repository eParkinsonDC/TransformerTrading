import torch
from torch.utils.data import Dataset


class ForexDataset(Dataset):
    """
    A PyTorch Dataset for preparing sequential forex data
    for time series forecasting.
    Args:
        1. data (np.ndarray): Numpy array of shape [num_samples, num_features]
            containing the dataset.
        2. seq_length (int, optional): Number of timesteps in
            each input sequence. Default is 60.
        3. prediction_length (int, optional): Number of future steps
            to predict. Default is 1.
        4. feature_dim (int, optional): Total number of features in the data
            (for dimension checking). Default is 4.
        5. target_column_idx (int, optional): Index of the column
            to use as the target variable (e.g., close=3). Default is 3.
    Attributes:
        1. data (np.ndarray): The input data array.
        2. seq_length (int): Length of the input sequence.
        3. pred_length (int): Number of future steps to predict.
        4. feature_dim (int): Number of features in the data.
        5. target_column_idx (int): Index of the target column.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a tuple (x, y) where x is the input sequence and y is the target sequence for prediction.
    """

    def __init__(
        self,
        data,
        seq_length=60,
        prediction_length=1,
        feature_dim=4,
        target_column_idx=3,
    ):

        self.data = data
        self.seq_length = seq_length
        self.pred_length = prediction_length
        self.feature_dim = feature_dim
        self.target_column_idx = target_column_idx

    def __len__(self):
        # The maximum starting index (MaxStartIndex) is:
        # MaxStartIndex = total_length - seq_length - prediction_length
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx: idx + self.seq_length]
        # Future price(s)
        y = self.data[
            idx + self.seq_length: idx + self.seq_length + self.pred_length,
            self.target_column_idx,
        ]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )
