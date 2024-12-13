from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input size = number of features fed into the model.
        # Each track (left and right) has 10 points, each with 2 coordinates.
        # This means the total input size is (n_track * 2) * 2
        input_size = n_track * 2 * 2

        # Output size = n_waypoints * 2 because each waypoint has 2 coordinates
        output_size = n_waypoints * 2

        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate the left and right boundaries of the track along the last dimension
        boundaries = torch.cat([track_left, track_right], dim=-1)
    
        # Flatten boundaries to feed into the model
        boundaries_flat = boundaries.view(boundaries.size(0), -1)
    
        # Pass the flattened boundaries through the model
        waypoints_flat = self.model(boundaries_flat)

        # Reshape the waypoints to the correct output shape
        waypoints = waypoints_flat.view(-1, self.n_waypoints, 2)

        return waypoints

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 4,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Encode the 2D track boundary data (left and right track) into a vector of size d_model.
        # It transforms the input data from shape (batch_size, n_track, 2) to (batch_size, n_track, d_model)
        self.track_encoder = nn.Linear(2, d_model)

        # An embedding layer for the waypoints. 
        # The model will learn a fixed embedding for each of the 3 waypoints, with each embedding being a vector of size d_model.
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Defines a single layer of the transformer decoder.
        # The parameters are:
        # d_model: The dimensionality of input and output vectors (64).
        # nhead: The number of attention heads for multi-head attention (2).
        # dim_feedforward: The size of the feedforward network (set to d_model * 2, i.e., 128).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 2, 
        )

        # Stacks multiple transformer decoder layers.
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # A linear layer that maps the transformer output (of shape d_model) back to 2D waypoints. 
        # The output size is 2 because each waypoint is represented by two values (x and y coordinates).
        self.output_head = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Batch size is the same for both left and right track boundaries
        batch_size = track_left.size(0)

        # Get embedding for left and right track in d_model space.
        # shape: (batch_size, n_track, 2) -> (batch_size, n_track, d_model)
        track_left_encoded = self.track_encoder(track_left)
        track_right_encoded = self.track_encoder(track_right)

        track_features = torch.cat([track_left_encoded, track_right_encoded], dim=1)

        # Transpose to shape (n_track, batch_size, d_model) for decoder
        track_features = track_features.permute(1, 0, 2)

        # Extracts learned query embeddings for the waypoints
        waypoint_queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Pass queries through decoder. Output shape: (n_waypoints, batch_size, d_model)
        attended_waypoints = self.transformer_decoder(waypoint_queries, track_features)

        # Transpose back to (batch_size, n_waypoints, d_model)
        attended_waypoints = attended_waypoints.permute(1, 0, 2)

        # Map the transformer output to 2D waypoints
        waypoints = self.output_head(attended_waypoints)

        return waypoints

class CNNPlanner(torch.nn.Module):

    # Block class to build the CNN and to make it a residual network.
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()
            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else torch.nn.Identity()
        
        def forward(self, x):
            res = self.skip(x)
            x = self.relu(self.bn1(self.c1(x)))
            x = self.relu(self.bn2(self.c2(x)))
            return self.relu(x + res)
        

    def __init__(
        self,
        n_waypoints: int = 3,
        channels: int = 64,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        # Normalize input image
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        channels = channels

        self.cnn = torch.nn.Sequential(
            self.Block(3, channels, stride=2),
            self.Block(channels, channels * 2, stride=2)
        )

        # Fully connected layer to predict waypoints
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 24 * 32, n_waypoints * 2),
        )


    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.cnn(x)
        x = self.fc(x)

        return x.view(x.size(0), self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
