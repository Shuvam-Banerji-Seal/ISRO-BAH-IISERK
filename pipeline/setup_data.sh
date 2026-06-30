#!/usr/bin/env bash
# Setup data symlinks for the pipeline.
# Run this once after cloning the repo, pointing at your raw data location.
#
# Usage:
#   ./setup_data.sh /path/to/data/root
#
# Example:
#   ./setup_data.sh /home/user/Isro/data
#
# The data root should contain raw/ and processed/ subdirectories.
# Processed NPZ files (master_dataset_*, stage1_*) are bundled in pipeline/data/processed/.

set -euo pipefail

DATA_ROOT="${1:-}"
if [ -z "$DATA_ROOT" ]; then
    echo "Usage: $0 /path/to/data/root"
    echo "Example: $0 /home/user/Isro/data"
    exit 1
fi

RAW_SRC="$DATA_ROOT/raw"
if [ ! -d "$RAW_SRC" ]; then
    echo "ERROR: $RAW_SRC does not exist"
    exit 1
fi

cd "$(dirname "$0")/data"
rm -f raw
ln -snf "$RAW_SRC" raw
echo "Symlinked: data/raw -> $RAW_SRC"
ls -la raw
