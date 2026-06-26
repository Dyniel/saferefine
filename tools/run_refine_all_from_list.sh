#!/usr/bin/env bash

ROOT="/home/student2/jaskra"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

LIST_FILE="${LIST_FILE:-${ROOT}/_refine_run_dirs.txt}"
[ -f "${LIST_FILE}" ] || { echo "[err] missing LIST_FILE=${LIST_FILE}"; exit 1; }

GPUS="${GPUS:-5 4 3 2}"

WORKERS="${WORKERS:-2}"   # dataloader workers per process
BS="${BS:-1}"             # batch size
CRF_GRID="${CRF_GRID:-1}"
DO_MORPH="${DO_MORPH:-1}"
DO_CRF="${DO_CRF:-1}"
DO_LEARNED_NOGRAPH="${DO_LEARNED_NOGRAPH:-1}"
DO_EXTRA_REFINERS_ON_GRAPH="${DO_EXTRA_REFINERS_ON_GRAPH:-0}"  # 0 = learned only for no_graph

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-${ROOT}/runs/_refine_all_${STAMP}}"
LOGDIR="${OUTROOT}/logs"
mkdir -p "${LOGDIR}"

RESULTS="${OUTROOT}/results.tsv"
QUEUE="${OUTROOT}/_queue.fifo"

echo "[cfg] LIST_FILE=${LIST_FILE}"
echo "[cfg] OUTROOT=${OUTROOT}"
echo "[cfg] GPUS=${GPUS}"
echo "[cfg] WORKERS=${WORKERS} BS=${BS}"
echo "[cfg] DO_MORPH=${DO_MORPH} DO_CRF=${DO_CRF} DO_LEARNED_NOGRAPH=${DO_LEARNED_NOGRAPH} EXTRA_ON_GRAPH=${DO_EXTRA_REFINERS_ON_GRAPH}"
echo

printf "run_dir\trun\tvariant\tstatus\traw_CpD\traw_C\traw_D\traw_sec_img\tmorph_CpD\tmorph_C\tmorph_D\tmorph_sec_img\tcrf_CpD\tcrf_C\tcrf_D\tcrf_sec_img\tlearned_CpD\tlearned_C\tlearned_D\tlearned_sec_img\tlog\n" > "${RESULTS}"

rm -f "${QUEUE}"
mkfifo "${QUEUE}"

parse_log_to_row() {
  local run_dir="$1"
  local log="$2"

  local run variant
  run="$(basename "$run_dir")"
  variant="unknown"
  if echo "$run" | grep -q "no_graph"; then variant="no_graph"; fi
  if echo "$run" | grep -q "dynk"; then variant="dyn"; fi
  if echo "$run" | grep -q "grid_only"; then variant="grid"; fi

  local status="OK"
  local rawCpD="-" rawC="-" rawD="-" rawSec="-"
  local mCpD="-" mC="-" mD="-" mSec="-"
  local cCpD="-" cC="-" cD="-" cSec="-"
  local lCpD="-" lC="-" lD="-" lSec="-"

  if grep -q "\[FAIL\]\|\[FATAL\]\|Traceback" "$log"; then
    status="FAIL"
  fi

  local line
  line="$(grep -E '^RAW[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
  if [ -n "$line" ]; then
    rawCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')"
    rawC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')"
    rawD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')"
    rawSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"
  fi

  line="$(grep -E '^MORPH[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
  if [ -n "$line" ]; then
    mCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')"
    mC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')"
    mD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')"
    mSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"
  fi

  line="$(grep -E '^CRF[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
  if [ -n "$line" ]; then
    cCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')"
    cC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')"
    cD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')"
    cSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"
  fi

  if grep -q '^LEARNED[[:space:]]+SKIP' "$log"; then
    lCpD="SKIP"
  else
    line="$(grep -E '^LEARNED[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
    if [ -n "$line" ]; then
      lCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')"
      lC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')"
      lD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')"
      lSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"
    fi
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_dir" "$run" "$variant" "$status" \
    "$rawCpD" "$rawC" "$rawD" "$rawSec" \
    "$mCpD" "$mC" "$mD" "$mSec" \
    "$cCpD" "$cC" "$cD" "$cSec" \
    "$lCpD" "$lC" "$lD" "$lSec" \
    "$log"
}

worker_loop() {
  local gpu="$1"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2

  local run_dir run log learned rc row
  while read -r run_dir; do
    [ -n "$run_dir" ] || continue
    run="$(basename "$run_dir")"
    log="${LOGDIR}/${run}.log"

    learned=0
    if echo "$run" | grep -q "no_graph"; then
      [ "${DO_LEARNED_NOGRAPH}" = "1" ] && learned=1
    else
      [ "${DO_EXTRA_REFINERS_ON_GRAPH}" = "1" ] && learned=1
    fi

    echo "[${gpu}] $(date '+%F %T') START ${run}" | tee -a "$log"
    echo "[${gpu}] run_dir=${run_dir}" | tee -a "$log"

    cmd=(
      python "${ROOT}/tools/refine_fair_eval.py"
      --run_dir "${run_dir}"
      --device cuda
      --workers "${WORKERS}"
      --batch "${BS}"
      --morph "${DO_MORPH}"
      --crf "${DO_CRF}"
      --crf_grid "${CRF_GRID}"
      --learned "${learned}"
    )

    echo "[${gpu}] CMD: ${cmd[*]}" | tee -a "$log"
    "${cmd[@]}" >> "$log" 2>&1
    rc=$?

    echo "[${gpu}] $(date '+%F %T') DONE rc=${rc} ${run}" | tee -a "$log"
    row="$(parse_log_to_row "$run_dir" "$log")"
    echo "$row" >> "${RESULTS}"
    echo "[${gpu}] wrote row -> results.tsv" | tee -a "$log"
    echo | tee -a "$log"
  done < "${QUEUE}"
}

echo "[info] feeding queue from: ${LIST_FILE}"
echo "[info] N=$(wc -l < "${LIST_FILE}" | tr -d ' ')"
echo

PIDS=()
for g in ${GPUS}; do
  worker_loop "$g" &
  PIDS+=("$!")
done

cat "${LIST_FILE}" > "${QUEUE}"

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo
echo "[OK] all workers finished."
echo "[out] results: ${RESULTS}"
echo "[out] logs:    ${LOGDIR}"
echo
echo "Tip:"
echo "  column -ts \$'\\t' \"${RESULTS}\" | less -S"
