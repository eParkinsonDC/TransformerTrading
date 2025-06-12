import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size=9,
        num_layers=2,
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        seq_length=30,
        prediction_length=1,
    ):
        super(TimeSeriesTransformer, self).__init__()

        # We embed each feature vector (feature_size) into a d_model-sized vector
        self.input_fc = nn.Linear(feature_size, d_model)
        self.input_dropout = nn.Dropout(dropout)  # Additional dropout

        # Positional Encoding (simple learnable or sinusoidal). We'll do a learnable here:
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final output: we want to forecast `prediction_length` steps for 1
        # dimension (Close price).
        # So we need a linear layer that maps d_model to prediction_length
        # This is the output layer that will give us the forecasted values
        # Note: prediction_length is the number of future steps we want to
        # predict
        # For example, if we want to predict the next 1 step,
        # prediction_length=1
        # If we want to predict the next 5 steps, prediction_length=5
        # The output will be of shape [batch_size, prediction_length]
        # If we want to predict multiple future steps, we can adjust
        # prediction_length accordingly
        # This layer will take the last output of the transformer and produce
        # the forecast
        # For example, if d_model=64 and prediction_length=1, the output will
        # be of shape [batch_size, 1]
        # If prediction_length=5, the output will be of shape [batch_size, 5]
        # This is the final layer that will output the forecasted values

        self.fc_out = nn.Linear(d_model, prediction_length)

    def forward(self, src):
        """
        src shape: [batch_size, seq_length, feature_size]
        """
        # batch_size, seq_len, _ = src.shape

        # First project features into d_model
        src = self.input_fc(src)  # -> [batch_size, seq_length, d_model]
        src = self.input_dropout(src)  # Apply dropout after input projection

        # Add positional embedding
        # pos_embedding -> [1, seq_length, d_model], so broadcast along batch dimension
        src = src + self.pos_embedding[:, :src.shape[1], :]

        # Transformer expects shape: [sequence_length, batch_size, d_model]
        # src = src.permute(1, 0, 2)  # -> [seq_length, batch_size, d_model] -> No need to do this because  PyTorch (since 1.12/1.13+) can use “nested tensors” for better speed if batch_first=True.
        # No permute needed when batch_first=True!
        encoded = self.transformer_encoder(src)
        # Pass through the transformer


        # We only want the output at the last time step for forecasting the future
        last_step = encoded[:, -1, :]  # [batch_size, d_model]
        out = self.fc_out(last_step)  # [batch_size, prediction_length]
        return out  # [batch_size, d_model]
