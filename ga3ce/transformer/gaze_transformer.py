import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeBiasedCrossAttention(nn.Module):
    """
    Multi-head cross-attention with an additive gaze-cone bias.

    Query: subject features
    Keys/values: encoded object features
    Bias: cosine similarity between view direction and object position
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=2,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_projection = nn.Linear(
            hidden_dim, hidden_dim
        )
        self.key_projection = nn.Linear(
            hidden_dim, hidden_dim
        )
        self.value_projection = nn.Linear(
            hidden_dim, hidden_dim
        )
        self.output_projection = nn.Linear(
            hidden_dim, hidden_dim
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        memory,
        attention_bias,
        key_padding_mask=None,
    ):
        batch_size, query_length, _ = query.shape
        _, memory_length, _ = memory.shape

        q = self.query_projection(query)
        k = self.key_projection(memory)
        v = self.value_projection(memory)

        # (B, sequence_length, hidden_dim)
        # -> (B, heads, sequence_length, head_dim)
        q = q.reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            batch_size,
            memory_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            batch_size,
            memory_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Standard scaled dot-product attention scores.
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # Apply the gaze-cone bias to the attention logits.
        # Bias shape: (B, 1, N) -> (B, 1, 1, N).
        scores = scores + attention_bias.unsqueeze(1)

        if key_padding_mask is not None:
            # True means the corresponding object is padding.
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        attention_weights = torch.softmax(
            scores, dim=-1
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        attended = torch.matmul(
            attention_weights, v
        )

        attended = attended.transpose(1, 2).contiguous()

        attended = attended.reshape(
            batch_size,
            query_length,
            self.hidden_dim,
        )

        output = self.output_projection(attended)

        return self.output_dropout(output)


class GazeDecoderLayer(nn.Module):
    """
    Transformer decoder layer with gaze-biased cross-attention.

    The subject sequence contains one token, so target self-attention
    operates on that single token.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=2,
        feedforward_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attention = GazeBiasedCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.linear1 = nn.Linear(
            hidden_dim, feedforward_dim
        )
        self.linear2 = nn.Linear(
            feedforward_dim, hidden_dim
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        subject,
        memory,
        attention_bias,
        object_padding_mask=None,
    ):
        # Target self-attention.
        attended_subject, _ = self.self_attention(
            subject,
            subject,
            subject,
            need_weights=False,
        )

        subject = self.norm1(
            subject + self.dropout1(attended_subject)
        )

        # Cross-attention with the actual gaze-cone bias.
        attended_objects = self.cross_attention(
            query=subject,
            memory=memory,
            attention_bias=attention_bias,
            key_padding_mask=object_padding_mask,
        )

        subject = self.norm2(
            subject + self.dropout2(attended_objects)
        )

        # Position-wise feed-forward network.
        feedforward = self.linear2(
            self.dropout(F.relu(self.linear1(subject)))
        )

        subject = self.norm3(
            subject + self.dropout3(feedforward)
        )

        return subject


class GazeTransformer(nn.Module):
    """
    GA3CE-style transformer prototype.

    Expected inputs:
        object_features: (B, N, 256)
        subject_features: (B, 362) or (B, 1, 362)
        view_directions: (B, 3)
        object_positions: (B, N, 3)
        object_padding_mask: optional (B, N), True means padding

    This model has randomly initialized weights. It is not a trained
    GA3CE gaze estimator.
    """

    def __init__(
        self,
        object_dim=256,
        subject_dim=362,
        hidden_dim=256,
        num_heads=2,
        num_encoder_layers=3,
        num_decoder_layers=3,
        feedforward_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.object_projection = nn.Linear(
            object_dim, hidden_dim
        )

        self.subject_projection = nn.Linear(
            subject_dim, hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.object_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.gaze_decoder = nn.ModuleList(
            [
                GazeDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.residual_gaze_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    @staticmethod
    def gaze_cone_bias(
        view_directions,
        object_positions,
        eps=1e-8,
    ):
        """
        Compute cosine similarity between each view direction
        and each object position.

        Inputs:
            view_directions: (B, 3)
            object_positions: (B, N, 3)

        Output:
            (B, 1, N)
        """
        view = F.normalize(
            view_directions, dim=-1, eps=eps
        )

        objects = F.normalize(
            object_positions, dim=-1, eps=eps
        )

        similarity = torch.sum(
            objects * view.unsqueeze(1),
            dim=-1,
        )

        return similarity.unsqueeze(1)

    def forward(
        self,
        object_features,
        subject_features,
        view_directions,
        object_positions,
        object_padding_mask=None,
    ):
        if object_features.ndim != 3:
            raise ValueError(
                "object_features must have shape (B, N, C)."
            )

        if subject_features.ndim == 2:
            subject_features = subject_features.unsqueeze(1)

        batch_size, num_objects, _ = object_features.shape

        if num_objects == 0:
            raise ValueError(
                "At least one object is required."
            )

        if subject_features.shape != (
            batch_size, 1, self.subject_projection.in_features
        ):
            raise ValueError(
                "subject_features must have shape (B, 362) "
                "or (B, 1, 362)."
            )

        if view_directions.shape != (batch_size, 3):
            raise ValueError(
                "view_directions must have shape (B, 3)."
            )

        if object_positions.shape != (
            batch_size, num_objects, 3
        ):
            raise ValueError(
                "object_positions must have shape (B, N, 3)."
            )

        if object_padding_mask is not None:
            if object_padding_mask.shape != (
                batch_size, num_objects
            ):
                raise ValueError(
                    "object_padding_mask must have shape (B, N)."
                )

            if object_padding_mask.all(dim=1).any():
                raise ValueError(
                    "Every sample must contain at least one valid object."
                )

        inputs = (
            object_features,
            subject_features,
            view_directions,
            object_positions,
        )

        if not all(
            torch.isfinite(tensor).all()
            for tensor in inputs
        ):
            raise ValueError(
                "Inputs contain non-finite values."
            )

        # Encode scene objects.
        object_embeddings = self.object_projection(
            object_features
        )

        encoded_objects = self.object_encoder(
            object_embeddings,
            src_key_padding_mask=object_padding_mask,
        )

        # Encode the subject query.
        subject = self.subject_projection(
            subject_features
        )

        # Compute the gaze-cone bias once for the object sequence.
        attention_bias = self.gaze_cone_bias(
            view_directions,
            object_positions,
        )

        # Every decoder layer applies the bias during cross-attention.
        for decoder_layer in self.gaze_decoder:
            subject = decoder_layer(
                subject=subject,
                memory=encoded_objects,
                attention_bias=attention_bias,
                object_padding_mask=object_padding_mask,
            )

        residual_gaze = self.residual_gaze_head(
            subject[:, 0]
        )

        return {
            "residual_gaze": residual_gaze,
            "encoded_objects": encoded_objects,
            "attention_bias": attention_bias,
        }