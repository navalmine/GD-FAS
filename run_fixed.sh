#!/usr/bin/env bash

set -euo pipefail

python GD-FAS.py --prompt_mode fixed --gs --temperature 0.1 --protocol O_C_to_M --save "$@"
python GD-FAS.py --prompt_mode fixed --gs --temperature 0.1 --protocol O_M_to_C --save "$@"
python GD-FAS.py --prompt_mode fixed --gs --temperature 1.0 --protocol C_M_to_O --save "$@"
