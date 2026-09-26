#!/usr/bin/env bash
# Evaluates the model from the last training run, found through the run
# manifest in the config's output.results_dir.
set -euo pipefail

# AI Runtime points $CODE_SOURCE_PATH at the first top-level entry of the
# extracted code: the repo itself for a `databricks air run` snapshot, but one
# of src/, jobs/, configs/ for the bundle's tarball, which extracts them side
# by side.
if [[ -f "$CODE_SOURCE_PATH/jobs/train.py" ]]; then
  cd "$CODE_SOURCE_PATH"
else
  cd "$(dirname "$CODE_SOURCE_PATH")"
fi
source jobs/air/pipeline.env

exec python jobs/evaluate.py --config_path "$CV_CONFIG_PATH"
