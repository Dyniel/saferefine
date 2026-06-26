#!/usr/bin/env bash

ROOT="/home/student2/jaskra"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

GPUS="${GPUS:-5 4 3 2}"
ROOTS="${ROOTS:-${ROOT}/runs/eccv_pub_segformer_deeplab ${ROOT}/runs/eccv_pub_segformer_amp0 ${ROOT}/runs/eccv_pt6 ${ROOT}/runs/eccv_closeout}"

MIN_EPOCH="${MIN_EPOCH:-2}"

REQUIRE_NPZ="${REQUIRE_NPZ:-${ROOT}/npz/REFUGE2_512_sanitized.npz}"
REQUIRE_IDX_VAL="${REQUIRE_IDX_VAL:-${ROOT}/idx_val.txt}"
REQUIRE_IDX_TRAIN="${REQUIRE_IDX_TRAIN:-}"

DO_MORPH="${DO_MORPH:-1}"
DO_CRF="${DO_CRF:-1}"
CRF_GRID="${CRF_GRID:-1}"
MORPH_KS="${MORPH_KS:-3,5}"
DO_LEARNED_NOGRAPH="${DO_LEARNED_NOGRAPH:-1}"

LIST_ONLY="${LIST_ONLY:-0}"   # 1 => tylko wypisz foldery i wyjdź

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-${ROOT}/runs/_refine_all_${STAMP}}"
LOGDIR="${OUTROOT}/logs"
mkdir -p "${LOGDIR}"

LIST="${OUTROOT}/run_dirs.txt"
RESULTS="${OUTROOT}/results.tsv"

echo "[cfg] OUTROOT=${OUTROOT}"
echo "[cfg] GPUS=${GPUS}"
echo "[cfg] ROOTS=${ROOTS}"
echo "[cfg] MIN_EPOCH=${MIN_EPOCH}"
echo "[cfg] REQUIRE_NPZ=${REQUIRE_NPZ}"
echo "[cfg] REQUIRE_IDX_VAL=${REQUIRE_IDX_VAL}"
echo "[cfg] REQUIRE_IDX_TRAIN=${REQUIRE_IDX_TRAIN:-<off>}"
echo "[cfg] DO_MORPH=${DO_MORPH} DO_CRF=${DO_CRF} CRF_GRID=${CRF_GRID} DO_LEARNED_NOGRAPH=${DO_LEARNED_NOGRAPH}"
echo

norm_path() {
  local p="$1"
  if [ -z "$p" ]; then echo ""; return 0; fi
  if [[ "$p" = /* ]]; then echo "$p"; else echo "${ROOT}/${p}"; fi
}

ckpt_meta_py() {
  # prints: epoch|npz_all|idx_val|idx_train
  local ckpt="$1"
  python - "$ckpt" <<'PY'
import sys, torch
p = sys.argv[1]
ck = torch.load(p, map_location="cpu")
args = ck.get("args", {}) or {}
ep = ck.get("epoch", -1)
try:
    ep = int(ep)
except Exception:
    ep = -1
npz = str(args.get("npz_all",""))
iv  = str(args.get("idx_val",""))
it  = str(args.get("idx_train",""))
print(f"{ep}|{npz}|{iv}|{it}")
PY
}

has_ep_ge_min() {
  local d="$1"
  # accept if there is any ep>=MIN_EPOCH file (ep002.pt etc)
  ls "$d/ckpt"/ep*.pt 2>/dev/null | grep -Eiq "ep0*${MIN_EPOCH}\.pt|ep0*[${MIN_EPOCH}-9]\.pt|ep[1-9][0-9]+\.pt" && return 0
  return 1
}

run_is_valid() {
  local d="$1"
  local run ckpt ep npz iv it nnpz niv nit

  [ -d "$d" ] || return 1
  run="$(basename "$d")"

  # drop debug garbage
  echo "$run" | grep -Eqi '(^_debug|debug)' && return 1

  ckpt="$d/ckpt/best.pt"
  [ -f "$ckpt" ] || return 1

  # epoch + meta
  if ! IFS='|' read -r ep npz iv it < <(ckpt_meta_py "$ckpt" 2>/dev/null); then
    return 1
  fi

  # epoch gate
  if ! [ "$ep" -ge "$MIN_EPOCH" ] 2>/dev/null; then
    has_ep_ge_min "$d" || return 1
  fi

  # normalize stored paths (they might be relative)
  nnpz="$(norm_path "$npz")"
  niv="$(norm_path "$iv")"
  nit="$(norm_path "$it")"

  if [ -n "$REQUIRE_NPZ" ] && [ "$nnpz" != "$REQUIRE_NPZ" ]; then return 1; fi
  if [ -n "$REQUIRE_IDX_VAL" ] && [ "$niv" != "$REQUIRE_IDX_VAL" ]; then return 1; fi
  if [ -n "$REQUIRE_IDX_TRAIN" ] && [ "$nit" != "$REQUIRE_IDX_TRAIN" ]; then return 1; fi

  return 0
}

: > "${LIST}"
for r in ${ROOTS}; do
  [ -d "$r" ] || continue
  while IFS= read -r d; do
    if run_is_valid "$d"; then
      echo "$d" >> "${LIST}"
    fi
  done < <(find "$r" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort)
done

N="$(wc -l < "${LIST}" | tr -d ' ')"
echo "[info] valid runs: ${N}"
echo "[info] list saved: ${LIST}"
echo

if [ "${N}" = "0" ] || [ -z "${N}" ]; then
  echo "[err] no VALID run dirs after gates"
  echo "[hint] najczęściej: w ckpt args są relatywne ścieżki inne niż ROOT, albo epoch<MIN_EPOCH"
  exit 1
fi

if [ "${LIST_ONLY}" = "1" ]; then
  cat "${LIST}"
  exit 0
fi

printf "run_dir\trun\tvariant\tstatus\traw_CpD\traw_C\traw_D\traw_sec_img\tmorph_CpD\tmorph_C\tmorph_D\tmorph_sec_img\tcrf_CpD\tcrf_C\tcrf_D\tcrf_sec_img\tlearned_CpD\tlearned_C\tlearned_D\tlearned_sec_img\tlog\n" > "${RESULTS}"

append_row_locked() {
  local row="$1"
  exec 9>>"${RESULTS}"
  flock 9
  echo -e "$row" >&9
  flock -u 9
  exec 9>&-
}

parse_log_to_row() {
  local run_dir="$1"
  local log="$2"
  local run variant status
  run="$(basename "$run_dir")"
  variant="unknown"
  echo "$run" | grep -q "no_graph" && variant="no_graph"
  echo "$run" | grep -q "dynk" && variant="dyn"
  echo "$run" | grep -q "grid_only" && variant="grid"

  status="OK"
  grep -q "\[FAIL\]\|\[FATAL\]\|Traceback\|error:" "$log" && status="FAIL"

  local rawCpD="-" rawC="-" rawD="-" rawSec="-"
  local mCpD="-" mC="-" mD="-" mSec="-"
  local cCpD="-" cC="-" cD="-" cSec="-"
  local lCpD="-" lC="-" lD="-" lSec="-"

  local line
  line="$(grep -E '^RAW[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
  [ -n "$line" ] && rawCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')" \
                 && rawC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')" \
                 && rawD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')" \
                 && rawSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"

  line="$(grep -E '^MORPH[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
  [ -n "$line" ] && mCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')" \
                 && mC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')" \
                 && mD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')" \
                 && mSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"

  line="$(grep -E '^CRF[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
  [ -n "$line" ] && cCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')" \
                 && cC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')" \
                 && cD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')" \
                 && cSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"

  if grep -q '^LEARNED[[:space:]]+SKIP' "$log"; then
    lCpD="SKIP"
  else
    line="$(grep -E '^LEARNED[[:space:]]+CpD=' "$log" | tail -n 1 || true)"
    [ -n "$line" ] && lCpD="$(echo "$line" | sed -n 's/.*CpD=\([0-9.]\+\).*/\1/p')" \
                   && lC="$(echo "$line" | sed -n 's/.*C=\([0-9.]\+\).*/\1/p')" \
                   && lD="$(echo "$line" | sed -n 's/.*D=\([0-9.]\+\).*/\1/p')" \
                   && lSec="$(echo "$line" | sed -n 's/.*sec\/img=\([0-9.]\+\).*/\1/p')"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
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

  while IFS= read -r run_dir; do
    [ -n "$run_dir" ] || continue
    local run log learned eval_k morph_ks crf_grid_arg rc row
    run="$(basename "$run_dir")"
    log="${LOGDIR}/${run}.log"

    learned=0
    if echo "$run" | grep -q "no_graph"; then
      [ "${DO_LEARNED_NOGRAPH}" = "1" ] && learned=1
    fi

    eval_k=16
    echo "$run" | grep -q "grid_only" && eval_k=0

    morph_ks="0"
    [ "${DO_MORPH}" = "1" ] && morph_ks="${MORPH_KS}"

    crf_grid_arg="0"
    [ "${DO_CRF}" = "1" ] && crf_grid_arg="${CRF_GRID}"

    echo "[${gpu}] $(date '+%F %T') START ${run}" | tee -a "$log"
    echo "[${gpu}] run_dir=${run_dir}" | tee -a "$log"

    cmd=(
      python "${ROOT}/tools/refine_fair_eval.py"
      --run_dir "${run_dir}"
      --device cuda
      --dyn_on_eval "feat"
      --eval_dyn_k "${eval_k}"
      --crf_grid "${crf_grid_arg}"
      --morph_ks "${morph_ks}"
      --learned "${learned}"
    )

    echo "[${gpu}] CMD: ${cmd[*]}" | tee -a "$log"
    "${cmd[@]}" >> "$log" 2>&1
    rc=$?

    echo "[${gpu}] $(date '+%F %T') DONE rc=${rc} ${run}" | tee -a "$log"
    row="$(parse_log_to_row "$run_dir" "$log")"
    append_row_locked "$row"
    echo "[${gpu}] wrote row -> results.tsv" | tee -a "$log"
    echo | tee -a "$log"
  done
}

# kolejka: prosty split listy na workerów przez round-robin (bez fifo)
# (fifo + multi-writer = ok, ale split jest prostszy i bardziej odporny)
mapfile -t RUNS < "${LIST}"
GARR=(${GPUS})
NG=${#GARR[@]}

# start workers z własnymi listami
PIDS=()
for i in "${!GARR[@]}"; do
  gpu="${GARR[$i]}"
  (
    for j in "${!RUNS[@]}"; do
      if [ $((j % NG)) -eq "$i" ]; then
        echo "${RUNS[$j]}"
      fi
    done | worker_loop "$gpu"
  ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo
echo "[OK] all workers finished."
echo "[out] results: ${RESULTS}"
echo "[out] logs:    ${LOGDIR}"
echo
echo "View:"
echo "  column -ts \$'\\t' \"${RESULTS}\" | less -S"
