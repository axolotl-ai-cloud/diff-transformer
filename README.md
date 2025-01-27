# Differential Transformer Implementation

Implementation of modeling code and conversion logic of `transformers` models from
non-differential to differential architecture based on the
[Differential Transformer paper](https://arxiv.org/abs/2410.05258).

## Installation

```shell
pip install git+https://github.com/axolotl-ai-cloud/diff-transformer.git
```

Editable:

```shell
git clone git@github.com:axolotl-ai-cloud/diff-transformer.git
cd diff-transformer
pip install -e .
```

## Usage

This is meant to be used as:

- `axolotl convert-diff-transformer path/to/config.yml`: Converts a `transformers`
model specified in axolotl config to

**Note:** The following will be set in the axolotl config output by the
`axolotl convert-diff-transformer` command.

```yaml
plugins:
  - axolotl_diff_transformer.plugin.DifferentialTransformerPlugin

diff_attention: true
```

Additional arguments include:

```yaml
# How often to log diffential attention-related metrics to wandb
# If not set, these metrics won't be logged
# Requires wandb logging to be enabled (see `wandb_project`, etc. yaml config options)
diff_attn_log_every: 100

# How many differential attention layers to monitor (strided from `0..k..num_layers`)
# Depends on `diff_attn_log_every` being set
diff_attn_num_monitor_layers: 3

# How many steps to "warmup" the mixing parameter for the negative component of differential attention
# Follows a linear warmup schedule from 0 to 1; if not specified, the mixing component is set to 1
diff_attn_warmup_steps: 1000
```

Additional command-line flags include:

```shell
# Prints debug information before and after model conversion
--debug

# Initializes negative attention component weights to zero
# Good for debugging output-preserving conversion of models
--zero-init

# Whether or not to apply sublayer normalization prior to output project calculation
# Also good for debugging output-preserving conversion of models
--sublayer-norm

# If provided, the given model's self attention heads will be split into one half
# positive heads and one half negative. If not, the number of heads will be doubled,
# and the second half will be used for the negative component of attention
# Note: This cannot be passed when the given model has an odd number of attention heads
--split-heads

# Whether to copy the positive component attention weights to the negative component
# This results in the converted model computing approximately the same function as the
# original
--mirror-weights
```
