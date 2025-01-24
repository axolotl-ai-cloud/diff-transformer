"""
Re-implemention of differential attention from the Differential Transformer paper
(https://arxiv.org/abs/2410.05258).
"""
# pylint: disable=invalid-name

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

try:
    from flash_attn.flash_attn_interface import flash_attn_func

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats key/value heads to match the number of query heads in multi-head attention.

    Args:
        x: Input tensor of shape `(batch_size, num_kv_heads, seq_len, head_dim)`.
        n_rep: Number of times to repeat each head.

    Returns:
        Tensor with repeated heads of shape `(batch_size, num_kv_heads * n_rep,
            seq_len, head_dim)`.
        If `n_rep` is 1, returns the input tensor unchanged.
    """
    batch_size, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(batch_size, n_kv_heads, n_rep, slen, head_dim)
        .reshape(batch_size, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth: int) -> float:
    """
    Lambda mixing parameter init function from the "Differential Transformer" paper.

    Args:
        depth: Index of layer to init lambda parameter.

    Returns:
        Lambda initialization value (decreasing with `depth`).
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class LlamaDifferentialAttentionBase(nn.Module):
    """
    Base class for differential attention implementations.

    This class implements the core differential attention mechanism used in Llama models.
    It supports both split heads and double projection modes for attention computation.
    """

    def __init__(self, config: Any, layer_idx: int):
        """
        Initializes the differential attention module.

        Args:
            config: Model configuration object containing hyperparameters, including:
                - hidden_size: The size of hidden states.
                - num_attention_heads: Number of attention heads.
                - num_key_value_heads: Number of key/value heads.
                - attention_bias: Whether to use bias in attention projections.
                - split_heads: Whether to use split heads mode.
                - rms_norm_eps: Epsilon for RMS normalization.
            layer_idx: The index of this layer in the model.

        Note:
            The initialization process consists of four steps:
            1. Configuration initialization (`_init_config`)
            2. Projection layers initialization (`_init_projections`)
            3. Differential parameters initialization (`_init_differential_params`)
            4. Normalization layers initialization (`_init_normalization`)
        """
        super().__init__()

        self.config = config
        self._init_config(layer_idx)
        self._init_projections()
        self._init_differential_params()
        self._init_normalization()

        # For logging
        self.attn1 = None
        self.attn2 = None
        self.lambda_full = None

    def _init_config(self, layer_idx: int) -> None:
        """
        Initializes configuration parameters for the attention layer. Sets up various
        dimension sizes and head counts based on the provided config. Handles both
        split heads and double projection modes.

        In split heads mode, the number of heads is divided by 2 (rounding down), which
        differs from the original implementation that required an even number.

        Args:
            layer_idx: Index of the current layer.
        """
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.base_num_heads = self.config.num_attention_heads
        self.base_num_kv_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.base_num_heads // self.base_num_kv_heads
        self.layer_idx = layer_idx

        if self.config.split_heads:
            self.heads_per_component = self.base_num_heads // 2
            self.kv_heads_per_component = self.base_num_kv_heads // 2
            self.value_head_dim = 2 * self.head_dim
        else:
            self.heads_per_component = self.base_num_heads
            self.kv_heads_per_component = self.base_num_kv_heads
            self.value_head_dim = self.head_dim

    def _init_projections(self) -> None:
        """
        Initializes the query, key, value, and output projection layers.

        Creates linear transformations for Q, K, V projections with dimensions
        depending on whether split heads or double projection mode is used.
        The output projection combines the attention heads back to model dimension.
        """
        if self.config.split_heads:
            q_out_dim = self.config.hidden_size
            k_out_dim = self.head_dim * self.base_num_kv_heads
        else:
            q_out_dim = self.config.hidden_size * 2
            k_out_dim = self.head_dim * self.base_num_kv_heads * 2

        self.q_proj = nn.Linear(
            self.config.hidden_size, q_out_dim, bias=self.config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size, k_out_dim, bias=self.config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.head_dim * self.base_num_kv_heads,
            bias=self.config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.base_num_heads * self.head_dim,
            self.config.hidden_size,
            bias=self.config.attention_bias,
        )

    def _init_differential_params(self) -> None:
        """
        Initializes parameters specific to differential attention.

        Creates learnable parameters for the differential attention mechanism:
        - Mixing parameter for negative attention component warmup phase.
        - Lambda parameters for queries and keys.
        - Initial lambda value based on layer index.
        - Rotary position embedding layer.
        """
        self.diff_attn_mix = 1.0  # Default to full mixing

        self.lambda_init = nn.Parameter(
            torch.full((), lambda_init_fn(self.layer_idx)),
            requires_grad=False,
        )
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _init_normalization(self) -> None:
        """
        Initializes normalization layers for the attention mechanism.

        Sets up either RMS normalization or identity transformation based on config.
        The normalization is applied to the sublayer output if enabled.
        """
        sublayer_norm = getattr(self.config, "sublayer_norm", True)
        if sublayer_norm:
            self.subln = LlamaRMSNorm(self.value_head_dim, eps=self.config.rms_norm_eps)
        else:
            self.subln = nn.Identity()

    def _prepare_attention_inputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares input tensors for attention computation.

        Projects input hidden states to query, key, and value spaces, then reshapes
        them for multi-head attention processing.

        Args:
            hidden_states: Input tensor of shape `(batch_size, seq_len,
            hidden_size)`.

        Returns:
            tuple: Tuple containing:
                - q1: Positive attention query component
                - q2: Negative attention query component
                - k1: Positive attention key component
                - k2: Negative attention key component
                - v: Value tensor
        """
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        q1 = q1.view(bsz, q_len, self.heads_per_component, self.head_dim).transpose(
            1, 2
        )
        q2 = q2.view(bsz, q_len, self.heads_per_component, self.head_dim).transpose(
            1, 2
        )
        k1 = k1.view(bsz, q_len, self.kv_heads_per_component, self.head_dim).transpose(
            1, 2
        )
        k2 = k2.view(bsz, q_len, self.kv_heads_per_component, self.head_dim).transpose(
            1, 2
        )
        v = v.view(bsz, q_len, self.base_num_kv_heads, self.head_dim).transpose(1, 2)

        return q1, q2, k1, k2, v

    def _apply_rotary_embeddings(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Applies rotary positional embeddings to queries and keys.

        Args:
            q1: Positive attention query component.
            q2: Negative attention query component.
            k1: Positive attention key component.
            k2: Negative attention key component.
            position_ids: Token position indices.
            position_embeddings: Pre-computed rotary embeddings (cos, sin).

        Returns:
            tuple: Tuple containing:
                - q1: Positive attention query with positional encoding.
                - q2: Negative attention query with positional encoding.
                - k1: Positive attention key with positional encoding.
                - k2: Negative attention key with positional encoding.
                - cos: Cosine part of rotary embeddings.
                - sin: Sine part of rotary embeddings.
        """
        if position_embeddings is None:
            LOG.warning(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(q1, position_ids)
        else:
            cos, sin = position_embeddings

        q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
        q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)

        return q1, q2, k1, k2, cos, sin

    def _handle_cache(
        self,
        k1: torch.Tensor,
        k2: torch.Tensor,
        v: torch.Tensor,
        past_key_value: Cache | None,
        cache_kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles key-value caching for autoregressive generation and the repetition of
        key-value heads to match the number of query heads.

        Args:
            k1: Positive attention key component.
            k2: Negative attention key component.
            v: Value tensor.
            past_key_value: Cache object for storing previous key-value pairs.
            cache_kwargs: Additional arguments for cache handling.

        Returns:
            tuple: Tuple containing:
                - k1: Processed positive attention key component.
                - k2: Processed negative attention key component.
                - v: Processed value tensor.
        """
        if past_key_value is not None:
            k = torch.stack([k1, k2], dim=1)
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            k1, k2 = k.unbind(dim=1)

        k1 = repeat_kv(k1, self.num_key_value_groups)
        k2 = repeat_kv(k2, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        if self.config.split_heads:
            v = torch.cat(torch.chunk(v, 2, dim=1), dim=-1)

        return k1, k2, v

    def _compute_lambda(self, q1: torch.Tensor) -> torch.Tensor:
        """
        Computes lambda values for differential attention.

        The lambda value is computed as λ₁ - λ₂ + λ_init, where λ₁ and λ₂ are computed
        from the learned parameters. `diff_attn_mix` is multiplied through the result
        for negative attention component warmup phase (if applicable).

        Args:
            q1: Positive attention query component, used for type casting.

        Returns:
            Computed lambda value for differential attention.
        """
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q1)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        return self.diff_attn_mix * lambda_full

    def _process_attention_output(
        self, attn: torch.Tensor, bsz: int, q_len: int
    ) -> torch.Tensor:
        """
        Processes and projects the attention output. Applies sublayer normalization,
        scales by (1 - λ_init), and projects back to model dimension.

        Args:
            attn: Raw attention output.
            bsz: Batch size.
            q_len: Query sequence length.

        Returns:
            Processed attention output of shape (batch_size, seq_len, hidden_size)
        """
        attn = self.subln(attn)
        # NOTE: this may need to be added back in, but doesn't interact well with
        # `diff_attn_mix`, and doesn't allow us to preserve the original model output.
        # attn = attn * self.diff_attn_mix * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, q_len, self.config.hidden_size)

        return self.o_proj(attn)


class LlamaDifferentialAttention(LlamaDifferentialAttentionBase):
    """
    Standard implementation of differential attention.

    This class implements the standard differential attention mechanism using
    explicit matrix multiplications for the attention computation.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # pylint: disable=unused-argument
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Computes differential attention using standard matrix multiplication operations.

        Args:
            hidden_states: Input tensor containing sequence to attend to.
            attention_mask: Mask to avoid attention on padding tokens.
            position_ids: Indices of positions for positional embeddings.
            past_key_value: Cached key and value tensors for autoregressive decoding.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to use cached key/value states.
            cache_position: Position indices for cached states.
            position_embeddings: Pre-computed positional embeddings.
            **kwargs: Additional arguments passed to the forward call.

        Returns:
            tuple containing:
                - Output tensor after attention computation.
                - Attention weights if output_attentions is True, else None.
                - Updated key-value cache if use_cache is True, else None.
        """
        bsz, q_len, _ = hidden_states.size()
        q1, q2, k1, k2, v = self._prepare_attention_inputs(hidden_states)
        q1, q2, k1, k2, cos, sin = self._apply_rotary_embeddings(
            q1, q2, k1, k2, position_ids, position_embeddings
        )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k1, k2, v = self._handle_cache(k1, k2, v, past_key_value, cache_kwargs)

        # Standard attention computation
        attn1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k1.shape[-2]]
            attn1 = attn1 + causal_mask
            attn2 = attn2 + causal_mask

        attn1 = F.softmax(attn1, dim=-1, dtype=torch.float32).type_as(attn1)
        attn2 = F.softmax(attn2, dim=-1, dtype=torch.float32).type_as(attn2)

        dropout_p = self.config.attention_dropout if self.training else 0.0
        attn1 = F.dropout(attn1, p=dropout_p, training=self.training)
        attn2 = F.dropout(attn2, p=dropout_p, training=self.training)

        lambda_full = self._compute_lambda(q1)
        attn = torch.matmul(attn1, v) - lambda_full * torch.matmul(attn2, v)
        attn = self._process_attention_output(attn, bsz, q_len)

        # Save for logging
        self.attn1 = attn1
        self.attn2 = attn2
        self.lambda_full = lambda_full

        if output_attentions:
            attn_weights = attn1 - lambda_full * attn2
            attn_weights = attn_weights.view(bsz, self.heads_per_component, q_len, -1)
            return attn, attn_weights
        return attn, None


class LlamaDifferentialSdpaAttention(LlamaDifferentialAttentionBase):
    """
    SDPA-based implementation of differential attention.

    This class implements differential attention using PyTorch's scaled_dot_product_attention
    for improved performance on supported hardware.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Computes differential attention using PyTorch's scaled dot product attention.

        Args:
            hidden_states: Input tensor containing sequence to attend to.
            attention_mask: Mask to avoid attention on padding tokens.
            position_ids: Indices of positions for positional embeddings.
            past_key_value: Cached key and value tensors for autoregressive decoding.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to use cached key/value states.
            cache_position: Position indices for cached states.
            position_embeddings: Pre-computed positional embeddings.
            **kwargs: Additional arguments passed to the forward call.

        Returns:
            tuple containing:
                - Output tensor after attention computation.
                - None for attention weights (SDPA doesn't support output_attentions).
                - Updated key-value cache if use_cache is True, else None.
        """
        if output_attentions:
            LOG.warning(
                "LlamaDifferentialModel is using LlamaDifferentialSdpaAttention, but "
                + "`torch.nn.functional.scaled_dot_product_attention` does not support "
                + "`output_attentions=True`. Falling back to the eager attention implementation."
            )

            # pylint: disable=duplicate-code
            return LlamaDifferentialAttention.forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        q1, q2, k1, k2, v = self._prepare_attention_inputs(hidden_states)
        q1, q2, k1, k2, cos, sin = self._apply_rotary_embeddings(
            q1, q2, k1, k2, position_ids, position_embeddings
        )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k1, k2, v = self._handle_cache(k1, k2, v, past_key_value, cache_kwargs)

        # SDPA-specific attention computation
        causal_mask = (
            None if attention_mask is None else attention_mask[:, :, :, : k1.shape[-2]]
        )
        is_causal = attention_mask is None and q_len > 1
        dropout_p = self.config.attention_dropout if self.training else 0.0

        if q1.device.type == "cuda" and causal_mask is not None:
            q1, q2 = q1.contiguous(), q2.contiguous()
            k1, k2 = k1.contiguous(), k2.contiguous()
            v = v.contiguous()

        attn1 = F.scaled_dot_product_attention(
            q1, k1, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        attn2 = F.scaled_dot_product_attention(
            q2, k2, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=is_causal
        )

        lambda_full = self._compute_lambda(q1)
        attn = attn1 - lambda_full * attn2
        attn = self._process_attention_output(attn, bsz, q_len)

        # Save for logging
        self.attn1 = attn1
        self.attn2 = attn2
        self.lambda_full = lambda_full

        return attn, None


class LlamaDifferentialFlashAttention2(LlamaDifferentialAttentionBase):
    """
    Flash Attention 2-based implementation of differential attention.

    This class implements differential attention using Flash Attention 2 for maximum
    performance on supported hardware.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the Flash Attention 2 differential attention module.

        Args:
            *args: Positional arguments passed to parent class.
            **kwargs: Keyword arguments passed to parent class.

        Raises:
            ImportError: If flash-attn library is not installed.
        """
        if not FLASH_ATTENTION_AVAILABLE:
            raise ImportError(
                "LlamaDifferentialFlashAttention2 requires flash-attn library. "
                "Please install with `pip install flash-attn --no-build-isolation`"
            )

        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Computes differential attention using Flash Attention 2.

        Args:
            hidden_states: Input tensor containing sequence to attend to.
            attention_mask: Mask to avoid attention on padding tokens.
            position_ids: Indices of positions for positional embeddings.
            past_key_value: Cached key and value tensors for autoregressive decoding.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to use cached key/value states.
            cache_position: Position indices for cached states.
            position_embeddings: Pre-computed positional embeddings.
            **kwargs: Additional arguments passed to the forward call.

        Returns:
            tuple containing:
                - Output tensor after attention computation.
                - None for attention weights (Flash Attention doesn't support output_attentions).
                - Updated key-value cache if use_cache is True, else None.
        """
        if output_attentions:
            LOG.warning(
                "LlamaDifferentialModel is using LlamaDifferentialFlashAttention2, but "
                + "flash attenion does not support `output_attentions=True`. Falling back "
                + "to the eager attention implementation."
            )

            # pylint: disable=duplicate-code
            return LlamaDifferentialAttention.forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        q1, q2, k1, k2, v = self._prepare_attention_inputs(hidden_states)
        q1, q2, k1, k2, cos, sin = self._apply_rotary_embeddings(
            q1, q2, k1, k2, position_ids, position_embeddings
        )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k1, k2, v = self._handle_cache(k1, k2, v, past_key_value, cache_kwargs)

        # Flash Attention specific processing
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.config.attention_dropout if self.training else 0.0

        if self.config.split_heads:
            v1, v2 = v.chunk(2, dim=-1)
            attn11 = flash_attn_func(q1, k1, v1, dropout_p=dropout_p, causal=True)
            attn12 = flash_attn_func(q1, k1, v2, dropout_p=dropout_p, causal=True)
            attn1 = torch.cat([attn11, attn12], dim=-1)

            attn21 = flash_attn_func(q2, k2, v1, dropout_p=dropout_p, causal=True)
            attn22 = flash_attn_func(q2, k2, v2, dropout_p=dropout_p, causal=True)
            attn2 = torch.cat([attn21, attn22], dim=-1)
        else:
            attn1 = flash_attn_func(q1, k1, v, dropout_p=dropout_p, causal=True)
            attn2 = flash_attn_func(q2, k2, v, dropout_p=dropout_p, causal=True)

        attn1, attn2 = attn1.transpose(1, 2), attn2.transpose(1, 2)

        lambda_full = self._compute_lambda(q1)
        attn = attn1 - lambda_full * attn2
        attn = self._process_attention_output(attn, bsz, q_len)

        # Save for logging
        self.attn1 = attn1
        self.attn2 = attn2
        self.lambda_full = lambda_full

        return attn, None
