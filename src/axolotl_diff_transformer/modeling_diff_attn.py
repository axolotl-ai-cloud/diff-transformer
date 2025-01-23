"""
Modeling for differential transformers (https://arxiv.org/abs/2410.05258).

This module implements differential attention variants of the LLaMA model,
providing various attention implementations for improved performance.
"""

import logging

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

from .diff_attn import (
    LlamaDifferentialAttention,
    LlamaDifferentialFlashAttention2,
    LlamaDifferentialSdpaAttention,
)

logger = logging.getLogger(__name__)


class LlamaDifferentialConfig(LlamaConfig):
    """
    Configuration class for Differential LLaMA model.

    Extends the base LLaMA configuration with additional parameters for differential
    attention mechanisms.
    """

    model_type = "llama-differential"

    def __init__(
        self,
        split_heads: bool = False,
        sublayer_norm: bool = True,
        zero_init: bool = False,
        mirror_weights: bool = False,
        **kwargs,
    ):
        """
        Initialize differential LLaMA configuration.

        Args:
            split_heads: Whether to use split heads mode for attention computation.
            sublayer_norm: Whether to apply normalization to sublayers.
            zero_init: Whether to initialize new weights to zero.
            mirror_weights: Whether to copy the positive attention component weights to
                the negative attention component.
            **kwargs: Additional arguments passed to LlamaConfig.
        """
        super().__init__(**kwargs)
        self.split_heads = split_heads
        self.sublayer_norm = sublayer_norm
        self.zero_init = zero_init
        self.mirror_weights = mirror_weights
        self.architectures = ["LlamaDifferentialModel"]
        self._attn_implementations = {
            "eager": "differential_eager",
            "sdpa": "differential_sdpa",
            "flash_attention_2": "differential_flash_attention_2",
        }


class LlamaDifferentialModel(LlamaModel):
    """
    LlamaModel with differential attention.

    This class extends the base LLaMA model by replacing standard attention with
    differential attention mechanisms.
    """

    config_class = LlamaDifferentialConfig
    base_model_prefix = "llama_differential"

    def __init__(self, config: LlamaDifferentialConfig):
        """
        Initialize a differential LLaMA model.

        Args:
            config: Configuration object for the model.

        Raises:
            ValueError: If specified attention implementation is not supported.
        """
        super().__init__(config)

        # Handle attention implementation
        attn_impl = config._attn_implementation or "eager"
        if attn_impl in config._attn_implementations:
            attn_impl = config._attn_implementations[attn_impl]

        # Validate attention implementation
        valid_impls = [
            None,
            "differential_eager",
            "differential_sdpa",
            "differential_flash_attention_2",
        ]
        if attn_impl not in valid_impls:
            raise ValueError(f"Invalid attention implementation: {attn_impl}")

        # Replace standard attention with differential attention in each layer
        attn_classes = {
            "differential_eager": LlamaDifferentialAttention,
            "differential_sdpa": LlamaDifferentialSdpaAttention,
            "differential_flash_attention_2": LlamaDifferentialFlashAttention2,
        }
        attn_class = attn_classes.get(attn_impl, LlamaDifferentialAttention)

        for idx, layer in enumerate(self.layers):
            layer.self_attn = attn_class(config, idx)

    @classmethod
    # pylint: disable=protected-access
    def _autoset_attn_implementation(
        cls,
        config: LlamaDifferentialConfig,
        **kwargs,  # pylint: disable=unused-argument
    ) -> LlamaDifferentialConfig:
        """
        Automatically set the attention implementation based on config.

        Args:
            config: Model configuration object.
            **kwargs: Additional arguments (unused).

        Returns:
            Updated configuration object.

        Raises:
            ValueError: If specified attention implementation is not supported.
        """
        config._attn_implementation_autoset = True
        attn_implementation = getattr(config, "_attn_implementation", None)

        # Map standard types to differential types if mapping exists
        if attn_implementation in config._attn_implementations:
            config._attn_implementation = config._attn_implementations[
                attn_implementation
            ]
            return config

        # If no mapping, validate it's a valid differential type
        valid_impls = [
            None,
            "differential_eager",
            "differential_sdpa",
            "differential_flash_attention_2",
        ]
        if attn_implementation not in valid_impls:
            message = (
                f"Specified `attn_implementation={attn_implementation}` is not supported. "
                f"The only possible arguments are: {', '.join(repr(x) for x in valid_impls if x)}"
            )
            raise ValueError(message)

        return config

    @classmethod
    def from_llama(
        cls,
        model: LlamaModel | LlamaForCausalLM,
        config: LlamaDifferentialConfig | None = None,
    ) -> "LlamaDifferentialModel":
        """
        Convert a `LlamaModel` to use differential attention.

        Args:
            model: Base LLaMA model to convert.
            config: Configuration for differential attention. If `None`, created from
                base model config.

        Returns:
            Converted model with differential attention.

        Raises:
            ValueError: If number of heads is not even when using `split_heads` mode.
        """
        logger.info(f"Converting {type(model).__name__} to {cls.__name__}")

        # Handle LlamaForCausalLM
        if isinstance(model, LlamaForCausalLM):
            model = model.model

        if config is None:
            config = LlamaDifferentialConfig(**model.config.__dict__)
            logger.debug(f"Created config: {config}")

        # Validate head counts if using split heads mode
        if config.split_heads:
            if config.num_attention_heads % 2 != 0:
                raise ValueError(
                    f"Number of attention heads ({config.num_attention_heads}) must be even "
                    "when using split_heads=True"
                )
            if config.num_key_value_heads % 2 != 0:
                raise ValueError(
                    f"Number of key/value heads ({config.num_key_value_heads}) must be even "
                    "when using split_heads=True"
                )

        new_model = cls(config)

        # Copy all weights except attention
        logger.debug("Copying embeddings and norm")
        new_model.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
        new_model.norm.load_state_dict(model.norm.state_dict())

        logger.debug("Copying layer weights")
        for layer_idx, (new_layer, old_layer) in enumerate(
            zip(new_model.layers, model.layers)
        ):
            # Copy everything except attention weights
            new_layer.mlp.load_state_dict(old_layer.mlp.state_dict())
            new_layer.input_layernorm.load_state_dict(
                old_layer.input_layernorm.state_dict()
            )
            new_layer.post_attention_layernorm.load_state_dict(
                old_layer.post_attention_layernorm.state_dict()
            )

            # Handle attention weights
            new_layer.self_attn.v_proj.load_state_dict(
                old_layer.self_attn.v_proj.state_dict()
            )
            new_layer.self_attn.o_proj.load_state_dict(
                old_layer.self_attn.o_proj.state_dict()
            )

            # Get the original projection sizes
            old_q_size = old_layer.self_attn.q_proj.weight.size(0)
            old_k_size = old_layer.self_attn.k_proj.weight.size(0)

            if not config.split_heads:
                logger.debug(
                    f"Layer {layer_idx}: Copying Q/K projections with sizes {old_q_size}, {old_k_size}"
                )
                new_layer.self_attn.q_proj.weight.data[:old_q_size].copy_(
                    old_layer.self_attn.q_proj.weight.data
                )
                new_layer.self_attn.k_proj.weight.data[:old_k_size].copy_(
                    old_layer.self_attn.k_proj.weight.data
                )

                if config.zero_init:
                    logger.debug(f"Layer {layer_idx}: Zero initializing")
                    with torch.no_grad():
                        new_layer.self_attn.q_proj.weight.data[old_q_size:].zero_()
                        new_layer.self_attn.k_proj.weight.data[old_k_size:].zero_()
                        new_layer.self_attn.lambda_q1.zero_()
                        new_layer.self_attn.lambda_k1.zero_()
                        new_layer.self_attn.lambda_q2.zero_()
                        new_layer.self_attn.lambda_k2.zero_()
                        new_layer.self_attn.lambda_init.zero_()
                elif config.mirror_weights:
                    # Mirror weights for second component
                    new_layer.self_attn.q_proj.weight.data[old_q_size:].copy_(
                        old_layer.self_attn.q_proj.weight.data
                    )
                    new_layer.self_attn.k_proj.weight.data[old_k_size:].copy_(
                        old_layer.self_attn.k_proj.weight.data
                    )

        logger.info("Conversion complete")

        return new_model


class LlamaDifferentialForCausalLM(LlamaForCausalLM):
    """
    `LlamaForCausalLM` with differential attention.

    This class extends the base LLaMA causal language model by incorporating
    differential attention mechanisms.
    """

    config_class = LlamaDifferentialConfig
    base_model_prefix = "llama_differential"

    def __init__(self, config: LlamaDifferentialConfig):
        """
        Initialize a differential LLaMA model for causal language modeling.

        Args:
            config: Configuration object for the model.
        """
        super().__init__(config)
        self.model = LlamaDifferentialModel(config)

    @classmethod
    # pylint: disable=protected-access
    def _autoset_attn_implementation(
        cls,
        config: LlamaDifferentialConfig,
        **kwargs,  # pylint: disable=unused-argument
    ) -> LlamaDifferentialConfig:
        """
        Automatically set the attention implementation based on config.

        Args:
            config: Model configuration object.
            **kwargs: Additional arguments (unused).

        Returns:
            Updated configuration object.

        Raises:
            ValueError: If specified attention implementation is not supported.
        """
        config._attn_implementation_autoset = True
        attn_implementation = getattr(config, "_attn_implementation", None)

        # Map standard types to differential types if mapping exists
        if attn_implementation in config._attn_implementations:
            config._attn_implementation = config._attn_implementations[
                attn_implementation
            ]

            return config

        # If no mapping, validate it's a valid differential type
        valid_impls = [
            None,
            "differential_eager",
            "differential_sdpa",
            "differential_flash_attention_2",
        ]
        if attn_implementation not in valid_impls:
            message = (
                f"Specified `attn_implementation={attn_implementation}` is not supported. "
                f"The only possible arguments are: {', '.join(repr(x) for x in valid_impls if x)}"
            )
            raise ValueError(message)

        return config

    @classmethod
    def from_llama(
        cls, model: LlamaForCausalLM, config: LlamaDifferentialConfig | None = None
    ) -> "LlamaDifferentialForCausalLM":
        """
        Convert a `LlamaForCausalLM` to use differential attention.

        Args:
            model: Base LLaMA model to convert.
            config: Configuration for differential attention. If `None`, created from
                base model config.

        Returns:
            Converted model with differential attention.

        Raises:
            ValueError: If number of heads is not even when using `split_heads` mode.
        """
        if config is None:
            config = LlamaDifferentialConfig(**model.config.__dict__)

        # Validate head counts if using split heads mode
        if config.split_heads:
            if config.num_attention_heads % 2 != 0:
                raise ValueError(
                    f"Number of attention heads ({config.num_attention_heads}) must be even "
                    "when using split_heads=True"
                )
            if config.num_key_value_heads % 2 != 0:
                raise ValueError(
                    f"Number of key/value heads ({config.num_key_value_heads}) must be even "
                    "when using split_heads=True"
                )

        new_model = cls(config)
        new_model.model = LlamaDifferentialModel.from_llama(model.model, config)
        new_model.lm_head.load_state_dict(model.lm_head.state_dict())

        return new_model


def register_diff_attn() -> None:
    """
    Register differential attention components with the transformers library.

    This function registers the differential attention configurations and model classes
    with the Auto* classes from `transformers`, making them available through the
    standard model loading pipeline.
    """
    # Register configs
    AutoConfig.register("llama-differential", LlamaDifferentialConfig)

    # Register models
    AutoModel.register(LlamaDifferentialConfig, LlamaDifferentialModel)
    AutoModelForCausalLM.register(LlamaDifferentialConfig, LlamaDifferentialForCausalLM)

    from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES

    LLAMA_ATTENTION_CLASSES["differential_eager"] = LlamaDifferentialAttention
    LLAMA_ATTENTION_CLASSES["differential_sdpa"] = LlamaDifferentialSdpaAttention
    LLAMA_ATTENTION_CLASSES[
        "differential_flash_attention_2"
    ] = LlamaDifferentialFlashAttention2
