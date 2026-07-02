#!/usr/bin/env bash
#
# Copy only the LAST checkpoint (model + train_state) from each run directory
# to a new location, preserving the directory structure.
#
# For every run directory directly under SRC, the script finds the highest
# model_step<N>.pt step that ALSO has a matching train_state_step<N>.pt, then
# copies both files (plus latest.pt, if present) into DST/<run_name>/.
#
# Usage:
#   scripts/copy_last_ckpts.sh [SRC] [DST] [--dry-run]
#
# Defaults:
#   SRC = /scratch-shared/mivanov1/hnet/checkpoints/train/comparison/S
#   DST = ${SRC}_last

set -euo pipefail

SRC="/scratch-shared/mivanov1/hnet/checkpoints/train/comparison/S"
DST=""
DRY_RUN=0

positional=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) positional+=("$arg") ;;
    esac
done

[[ ${#positional[@]} -ge 1 ]] && SRC="${positional[0]}"
[[ ${#positional[@]} -ge 2 ]] && DST="${positional[1]}"
SRC="${SRC%/}"
[[ -z "$DST" ]] && DST="${SRC}_last"
DST="${DST%/}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source directory does not exist: $SRC" >&2
    exit 1
fi

echo "Source:      $SRC"
echo "Destination: $DST"
[[ $DRY_RUN -eq 1 ]] && echo "(dry run — nothing will be copied)"
echo

for run_dir in "$SRC"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name="$(basename "$run_dir")"

    # Highest step that has a model checkpoint.
    last_step=""
    for f in "$run_dir"model_step*.pt; do
        [[ -e "$f" ]] || continue
        step="${f##*model_step}"
        step="${step%.pt}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if [[ -z "$last_step" || "$step" -gt "$last_step" ]]; then
            # Require a matching train_state so the pair is complete.
            [[ -e "${run_dir}train_state_step${step}.pt" ]] && last_step="$step"
        fi
    done

    if [[ -z "$last_step" ]]; then
        echo "[$run_name] no complete (model+train_state) checkpoint found — skipping" >&2
        continue
    fi

    echo "[$run_name] last step = $last_step"
    dest_run="$DST/$run_name"

    files=("model_step${last_step}.pt" "train_state_step${last_step}.pt")
    [[ -e "${run_dir}latest.pt" ]] && files+=("latest.pt")

    if [[ $DRY_RUN -eq 1 ]]; then
        for fname in "${files[@]}"; do
            echo "    would copy $fname -> $dest_run/"
        done
        continue
    fi

    mkdir -p "$dest_run"
    for fname in "${files[@]}"; do
        echo "    copying $fname"
        cp -p "${run_dir}${fname}" "$dest_run/"
    done
done

echo
echo "Done."
