#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/retrain_if_sft.py "$@"
