"""Module for CLI arguments for convert-diff-transformer command."""

from dataclasses import dataclass, field


@dataclass
class ConvertDiffTransformerCliArgs:
    """Dataclass with arguments for convert-diff-transformer CLI"""

    debug: bool = field(default=False)
    zero_init: bool = field(default=False)
    sublayer_norm: bool = field(default=True)
    split_heads: bool = field(default=False)
    mirror_weights: bool = field(default=False)
