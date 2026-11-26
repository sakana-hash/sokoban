#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
batch=
start_from=
profile_enabled=true
train_args=()
while (( $# )); do
    case "$1" in
        --batch-dir|--start-from)
            if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
                echo "$1 需要一个参数" >&2
                exit 2
            fi
            if [[ $1 == --batch-dir ]]; then
                batch=$2
            else
                start_from=$2
            fi
            shift ;;
        --profile) profile_enabled=true ;;
        --no-profile) profile_enabled=false ;;
        *) train_args+=("$1") ;;
    esac
    shift
done
train_command=(python -m training.train)
if [[ $profile_enabled == true ]]; then
    train_command+=(--profile --profile-start 160000)
fi
# User-supplied sampling options override the defaults. All runs share the same budget.
train_args+=(--device cuda --model-version 3 --seed 42 --steps 400000)
train_command+=("${train_args[@]}")
if [[ -z $batch ]]; then
    mkdir -p runs
    batch=$(mktemp -d "runs/ddqn-v3-400k-$(date +%Y%m%d-%H%M%S)-XXXXXX")
elif [[ ! -d $batch ]]; then
    echo "找不到已有批次目录 $batch" >&2
    exit 2
fi
echo "本批实验目录 $batch"

run_experiment() {
    local name=$1 deadlock_flag=$2
    shift 2
    if [[ -n $start_from && $name != "$start_from" ]]; then
        return 0
    fi
    start_from=
    local output="$batch/$name"
    echo "开始实验 $name"
    "${train_command[@]}" --output "$output" "$deadlock_flag" "$@"
    local eval_args=(--device cuda --checkpoint "$output/best.pt" "$deadlock_flag" --compare-random)
    python -m training.evaluate "${eval_args[@]}" --generated 1000 --max-boxes 1 --seed 2027
    python -m training.evaluate "${eval_args[@]}" --seed 2026
}

# Original single-box curriculum, then the three experiments with annealed easy starts.
run_experiment baseline --no-deadlock-detection --no-small-board-curriculum --no-anneal-easy-starts --revisit-penalty 0
run_experiment mix --no-deadlock-detection --no-small-board-curriculum --revisit-penalty 0
run_experiment revisit --no-deadlock-detection --no-small-board-curriculum --revisit-penalty 0.02
run_experiment revisit-004 --no-deadlock-detection --no-small-board-curriculum --revisit-penalty 0.04

# The historical standard-boards run used the same settings as deadlock-neg1.
run_experiment deadlock-neg1 --deadlock-detection --no-small-board-curriculum --deadlock-penalty -1
run_experiment deadlock-neg10 --deadlock-detection --no-small-board-curriculum --deadlock-penalty -10
run_experiment small-boards --deadlock-detection --small-board-curriculum --deadlock-penalty -1

if [[ -n $start_from ]]; then
    echo "找不到实验 $start_from" >&2
    exit 2
fi
