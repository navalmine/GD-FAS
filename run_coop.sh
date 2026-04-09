#!/usr/bin/env bash

set -euo pipefail

python GD-FAS.py --prompt_mode coop --n_ctx 4 --gs --temperature 0.1 --protocol O_C_to_M "$@"
python GD-FAS.py --prompt_mode coop --n_ctx 4 --gs --temperature 0.1 --protocol O_M_to_C "$@"
python GD-FAS.py --prompt_mode coop --n_ctx 4 --gs --temperature 1.0 --protocol C_M_to_O "$@"
