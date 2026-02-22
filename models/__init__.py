"""Models package for sign metric learning."""

from models.mlp_encoder import MLPEncoder, build_mlp_encoder
from models.temporal_transformer import TemporalTransformerEncoder, build_transformer_encoder
from models.gcn_encoder import GCNEncoder, build_gcn_encoder
from models.prototypical import PrototypicalNetwork
from models.siamese import SiameseNetwork, MatchingNetwork


def build_encoder(cfg: dict, representation: str = "raw"):
    """Build an encoder based on config.

    Args:
        cfg: Full config dict.
        representation: ``'raw'``, ``'pairwise'``, or ``'graph'``.

    Returns:
        Encoder module.
    """
    encoder_name = cfg["model"]["encoder"]
    if encoder_name == "mlp":
        return build_mlp_encoder(cfg, representation)
    elif encoder_name == "transformer":
        return build_transformer_encoder(cfg, representation)
    elif encoder_name == "gcn":
        return build_gcn_encoder(cfg, representation)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


def build_few_shot_model(cfg: dict, encoder):
    """Build a few-shot model wrapping the given encoder.

    Args:
        cfg: Full config dict.
        encoder: Backbone encoder module.

    Returns:
        Few-shot model (PrototypicalNetwork, SiameseNetwork, or MatchingNetwork).
    """
    method = cfg["few_shot"]["method"]
    if method == "prototypical":
        return PrototypicalNetwork(encoder, distance="euclidean")
    elif method == "siamese":
        return SiameseNetwork(encoder, embedding_dim=cfg["model"]["embedding_dim"])
    elif method == "matching":
        return MatchingNetwork(encoder)
    else:
        raise ValueError(f"Unknown few-shot method: {method}")


__all__ = [
    "MLPEncoder", "build_mlp_encoder",
    "TemporalTransformerEncoder", "build_transformer_encoder",
    "GCNEncoder", "build_gcn_encoder",
    "PrototypicalNetwork", "SiameseNetwork", "MatchingNetwork",
    "build_encoder", "build_few_shot_model",
]
