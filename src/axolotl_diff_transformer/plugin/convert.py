"""Implementation of conversion logic from LLaMA to Diff-LLaMA architecture."""
import logging

import torch
from axolotl.logging_config import configure_logging
from torch import nn
from transformers.models.diffllama.modeling_diffllama import DiffLlamaConfig, DiffLlamaModel, DiffLlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

configure_logging()
LOG = logging.getLogger(__name__)


def convert_to_diffllama(
    model: LlamaModel | LlamaForCausalLM,
    config: DiffLlamaConfig | None = None,
    zero_init: bool = False,
    sublayer_norm: bool = True,
    split_heads: bool = False,
    mirror_weights: bool = False,
) -> DiffLlamaModel | DiffLlamaForCausalLM:
    """
    Convert a `LlamaModel` or `LlamaForCausalLM` to use differential attention.

    Args:
        model: Base LLaMA model to convert.
        config: Configuration for differential attention. If `None`, created from
            base model config.

    Returns:
        Converted model with differential attention.

    Raises:
        ValueError: If number of heads is not even when using `split_heads` mode.
    """
    if not isinstance(model, (LlamaModel, LlamaForCausalLM)):
        raise NotImplementedError("passed model type must be in (LlamaModel, LlamaForCausalLM)")

    # Handle LlamaForCausalLM
    if isinstance(model, LlamaForCausalLM):
        new_model = DiffLlamaForCausalLM(config)
        new_model.model = convert_to_diffllama(
            model.model, 
            config=config,
            zero_init=zero_init,
            sublayer_norm=sublayer_norm,
            split_heads=split_heads,
            mirror_weights=mirror_weights
        )
        new_model.lm_head.load_state_dict(model.lm_head.state_dict())
        return new_model

    # Create config with base model parameters
    if config is None:
        config = DiffLlamaConfig(**model.config.__dict__)

    # Handle split heads mode
    if split_heads:
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
        
        # Update head counts and dimensions for split heads
        config.num_attention_heads = config.num_attention_heads
        config.num_key_value_heads = config.num_key_value_heads
    else:
        config.num_attention_heads = config.num_attention_heads * 2
        config.num_key_value_heads = config.num_key_value_heads * 2

    # Store other conversion parameters
    config.split_heads = split_heads
    config.sublayer_norm = sublayer_norm
    config.zero_init = zero_init
    config.mirror_weights = mirror_weights

    # Create new model with updated config
    new_model = DiffLlamaModel(config)

    # Copy embeddings and norm
    new_model.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
    new_model.norm.load_state_dict(model.norm.state_dict())

    # Process each layer
    for layer_idx, (new_layer, old_layer) in enumerate(zip(new_model.layers, model.layers)):
        # Copy non-attention weights
        new_layer.mlp.load_state_dict(old_layer.mlp.state_dict())
        new_layer.input_layernorm.load_state_dict(old_layer.input_layernorm.state_dict())
        new_layer.post_attention_layernorm.load_state_dict(old_layer.post_attention_layernorm.state_dict())

        # Handle attention weights
        attn = new_layer.self_attn
        old_attn = old_layer.self_attn

        # Log original dimensions for first layer
        if layer_idx == 0:
            LOG.info(f"\nConfig:")
            LOG.info(f"\n{config}")

            LOG.info(f"\nOriginal attention dimensions (layer 0):")
            LOG.info(f"q_proj: {old_attn.q_proj.weight.shape}")
            LOG.info(f"k_proj: {old_attn.k_proj.weight.shape}")
            LOG.info(f"v_proj: {old_attn.v_proj.weight.shape}")
            LOG.info(f"o_proj: {old_attn.o_proj.weight.shape}")

            LOG.info(f"\nNew attention dimensions (layer 0):")
            LOG.info(f"q_proj: {attn.q_proj.weight.shape}")
            LOG.info(f"k_proj: {attn.k_proj.weight.shape}")
            LOG.info(f"v_proj: {attn.v_proj.weight.shape}")
            LOG.info(f"o_proj: {attn.o_proj.weight.shape}")
        
        if split_heads:
            # In split heads mode, copy all weights directly
            attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
            attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
        else:
            # For Q_proj: [576, 576] -> [1152, 576]
            old_q_data = old_attn.q_proj.weight.data 
            new_q_data = attn.q_proj.weight.data
            q_mid = new_q_data.size(0) // 2  # 576
            new_q_data[:q_mid].copy_(old_q_data)
            new_q_data[q_mid:].zero_()

            # For K_proj: [192, 576] -> [384, 576]
            old_k_data = old_attn.k_proj.weight.data
            new_k_data = attn.k_proj.weight.data 
            k_mid = new_k_data.size(0) // 2  # 192
            new_k_data[:k_mid].copy_(old_k_data)
            new_k_data[k_mid:].zero_()

            # # For V_proj
            # old_v_data = old_attn.v_proj.weight.data  # [192, 576]
            # new_v_data = attn.v_proj.weight.data      # [384, 576]
            # v_mid = new_v_data.size(0) // 2           # 192

            # # Calculate size based on the original num_key_value_heads
            # # Original model has 3 KV heads, which gets doubled to 6
            # groups_per_head = config.num_attention_heads // config.num_key_value_heads  # 3
            # kv_size = v_mid // groups_per_head  # 192/3 = 64

            # # First half - will become first half of head_dim after cat
            # new_v_data[:kv_size].copy_(old_v_data[:kv_size] / groups_per_head)
            # new_v_data[kv_size:v_mid].zero_()

            # # Second half - will become second half of head_dim after cat
            # new_v_data[v_mid:v_mid+kv_size].copy_(old_v_data[:kv_size] / groups_per_head)
            # new_v_data[v_mid+kv_size:].zero_()

            # # For O_proj
            # old_o_data = old_attn.o_proj.weight.data  # [576, 576]
            # new_o_data = attn.o_proj.weight.data      # [576, 1152]
            # o_mid = new_o_data.size(1) // 2           # 576
            # o_size = o_mid // groups_per_head         # 576/3 = 192

            # # Scale to compensate only for repeat_kv
            # new_o_data[:, :o_size].copy_(old_o_data[:, :o_size] * groups_per_head)
            # new_o_data[:, o_size:].zero_()

            # For V_proj
            old_v_data = old_attn.v_proj.weight.data  # [192, 576]
            new_v_data = attn.v_proj.weight.data      # [384, 576]
            v_mid = new_v_data.size(0) // 2           # 192
            groups_per_head = config.num_attention_heads // config.num_key_value_heads  # 3

            # Copy values that will end up in first half of head_dim after chunk+cat
            new_v_data[:v_mid].copy_(old_v_data)  # Original scale for first half
            new_v_data[v_mid:].zero_()  # Zero second half - this becomes second half of head_dim

            # For O_proj
            old_o_data = old_attn.o_proj.weight.data  # [576, 576]
            new_o_data = attn.o_proj.weight.data      # [576, 1152]
            o_mid = new_o_data.size(1) // 2           # 576

            # First half at original scale, second half zeroed
            new_o_data[:, :o_mid].copy_(old_o_data)
            new_o_data[:, o_mid:].zero_()

            # Handle lambda parameters for differential attention
            if zero_init:
                with torch.no_grad():
                    attn.lambda_q1.zero_()
                    attn.lambda_k1.zero_()
                    attn.lambda_q2.zero_()
                    attn.lambda_k2.zero_()

        # Handle sublayer normalization
        if not sublayer_norm:
            attn.groupnorm = nn.Identity()

    # import ipdb; ipdb.set_trace()

    del model
    return new_model
