from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256

@dataclass
class RoutingConfig:
    
    multiheaded: bool = False
    num_heads: int = 4
    window_size: int = 8
    d_similarity: int = -1
    random: bool = False
    compression_ratio: float = 1.0
    softmax_gating: bool = False
    identity_routing: bool = False
    entropy_routing: bool = False
    byte_vocab_size: int = 256
    learn_entropy_thresholds: bool = True
    single_projection: bool = False
    bm_head_cos_routing: bool = False

@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
    routing_cfg: RoutingConfig = field(default_factory=RoutingConfig)
    contrastive_loss: bool = False
    contrastive_loss_arch: str = "T2"
    contrastive_num_neg_samples: int = 8
