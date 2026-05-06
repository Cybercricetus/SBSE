#!/usr/bin/env bash
# Run the Track A experiment on every nrp-*.txt instance in data/.
#
# Usage:
#   ./run.sh                    # defaults: ratio=0.3, runs=30, evals=50000
#   RATIO=0.5 ./run.sh          # override cost ratio
#   RUNS=10 EVALS=20000 ./run.sh   # quicker run
#
# Outputs:
#   results/<name>_r<ratio>.json        per-run history
#   results/<name>_r<ratio>_*.png       convergence + box plots
#   results/run.log                     full stdout from every instance

set -euo pipefail

# ---- config (overridable via env) ----
RATIO="${RATIO:-0.3}"
RUNS="${RUNS:-30}"
EVALS="${EVALS:-50000}"
SEED="${SEED:-0}"

DATA_DIR="data"
RESULTS_DIR="results"
LOG_FILE="${RESULTS_DIR}/run.log"

mkdir -p "${RESULTS_DIR}"
: > "${LOG_FILE}"   # truncate

# Tag for filenames: 0.3 -> r03, 0.5 -> r05
ratio_tag="r$(printf '%s' "${RATIO}" | tr -d '.')"

# Sort lexicographically so e* runs before g* runs before m*
mapfile -t instances < <(ls "${DATA_DIR}"/nrp-*.txt 2>/dev/null | sort)

if [[ ${#instances[@]} -eq 0 ]]; then
    echo "No nrp-*.txt files found in ${DATA_DIR}/" >&2
    exit 1
fi

echo "Running ${#instances[@]} instances  |  ratio=${RATIO}  runs=${RUNS}  evals=${EVALS}"
echo "Logs: ${LOG_FILE}"
echo "----------------------------------------------------------------"

total_start=$(date +%s)

for inst in "${instances[@]}"; do
    name="$(basename "${inst}" .txt)"          # e.g. nrp-e1
    short="${name#nrp-}"                       # e.g. e1
    out_stem="${RESULTS_DIR}/${short}_${ratio_tag}"

    printf '[%-6s] ' "${short}"
    inst_start=$(date +%s)

    {
        echo
        echo "================================================================"
        echo "INSTANCE: ${name}    ratio=${RATIO}    runs=${RUNS}    evals=${EVALS}"
        echo "================================================================"
    } >> "${LOG_FILE}"

    if python main.py \
        --instance "${inst}" \
        --cost-ratio "${RATIO}" \
        --runs "${RUNS}" \
        --evals "${EVALS}" \
        --seed "${SEED}" \
        --output "${out_stem}.json" \
        --plot "${out_stem}" \
        >> "${LOG_FILE}" 2>&1
    then
        elapsed=$(( $(date +%s) - inst_start ))
        printf 'done in %3dm%02ds  -> %s.json\n' \
            $((elapsed / 60)) $((elapsed % 60)) "${out_stem}"
    else
        printf 'FAILED (see %s)\n' "${LOG_FILE}"
        exit 1
    fi
done

total_elapsed=$(( $(date +%s) - total_start ))
echo "----------------------------------------------------------------"
printf 'All %d instances done in %dm%02ds\n' \
    "${#instances[@]}" $((total_elapsed / 60)) $((total_elapsed % 60))
echo "Results in: ${RESULTS_DIR}/"
