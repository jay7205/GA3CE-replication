import torch
import torch.nn as nn


class D3PositionalEncoding(nn.Module):
    """
    Direction-Distance-Decomposed (D3) positional encoding.

    Implements the D3 decomposition described in GA3CE.

    The frequency configuration must be supplied explicitly.
    The paper's exact frequency configuration has not yet been
    verified, so this module does not invent one.
    """

    def __init__(
        self,
        direction_frequencies,
        distance_frequencies,
        eps=1e-8,
    ):
        super().__init__()

        direction_frequencies = torch.as_tensor(
            direction_frequencies,
            dtype=torch.float32,
        )

        distance_frequencies = torch.as_tensor(
            distance_frequencies,
            dtype=torch.float32,
        )

        if direction_frequencies.ndim != 1:
            raise ValueError(
                "Direction frequencies must be one-dimensional."
            )

        if distance_frequencies.ndim != 1:
            raise ValueError(
                "Distance frequencies must be one-dimensional."
            )

        if direction_frequencies.numel() == 0:
            raise ValueError(
                "At least one direction frequency is required."
            )

        if distance_frequencies.numel() == 0:
            raise ValueError(
                "At least one distance frequency is required."
            )

        if not torch.isfinite(direction_frequencies).all():
            raise ValueError("Invalid direction frequencies.")

        if not torch.isfinite(distance_frequencies).all():
            raise ValueError("Invalid distance frequencies.")

        self.register_buffer(
            "direction_frequencies",
            direction_frequencies,
        )

        self.register_buffer(
            "distance_frequencies",
            distance_frequencies,
        )

        self.eps = eps

    @staticmethod
    def sinusoidal_encoding(values, frequencies):
        """
        Encode the final coordinate dimension using explicit
        sinusoidal frequencies.

        Output shape:
            (..., coordinates, frequencies, 2)
        """
        angles = (
            values.unsqueeze(-1)
            * frequencies
        )

        return torch.stack(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1,
        ).flatten(start_dim=-2)

    def forward(self, points):
        """
        Args:
            points: Tensor shaped (..., 3).

        Returns:
            Dictionary containing direction, distance and
            concatenated D3 features.

        Note:
            This is a configurable encoding implementation.
            Exact GA3CE frequency reproduction is pending verification.
        """
        if points.shape[-1] != 3:
            raise ValueError(
                f"Expected points with final dimension 3, "
                f"got {tuple(points.shape)}"
            )

        if not torch.isfinite(points).all():
            raise ValueError("Points contain non-finite values.")

        distance = torch.linalg.vector_norm(
            points,
            dim=-1,
            keepdim=True,
        )

        direction = points / distance.clamp_min(self.eps)

        direction_features = self.sinusoidal_encoding(
            direction,
            self.direction_frequencies,
        ).flatten(start_dim=-2)

        distance_features = self.sinusoidal_encoding(
            distance,
            self.distance_frequencies,
        ).flatten(start_dim=-2)

        d3_features = torch.cat(
            [direction_features, distance_features],
            dim=-1,
        )

        return {
            "direction": direction,
            "distance": distance,
            "direction_features": direction_features,
            "distance_features": distance_features,
            "d3_features": d3_features,
        }