#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__"

pytest -v --durations=10 -n8 /workspace/diff-transformer/cicd_tests/
