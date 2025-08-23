if [ -z "${DATAROOT}" ]; then
    echo "DATAROOT is not set."
    exit 1
else
    echo "DATAROOT: ${DATAROOT}"
fi
NUM_SAMPLES=128
REMOVE_NEWLINE_TAB=false
STOP_WORDS=""

if [ -z "${STOP_WORDS}" ]; then
    STOP_WORDS=""
else
    STOP_WORDS="--stop_words \"${STOP_WORDS}\""
fi

if [ "${REMOVE_NEWLINE_TAB}" = false ]; then
    REMOVE_NEWLINE_TAB=""
else
    REMOVE_NEWLINE_TAB="--remove_newline_tab"
fi

# task name in `synthetic.yaml`
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "fwe"
    "qa_1"
)

SEQ_LENGTHS=(
    8192
    16384
    32768
    65536
    131072
    262144
    524288
)


MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" ###
MODEL_TEMPLATE_TYPE="base"
TOKENIZER_PATH=$MODEL_PATH
TOKENIZER_TYPE="hf"


BENCHMARK=synthetic
declare -n TASKS=$BENCHMARK

echo "TASKS: ${TASKS[@]}"
PIDS=()
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        python -u ruler_data_prepare.py \
            --data_dir ${DATAROOT} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            --chunk_amount 8 \
            ${REMOVE_NEWLINE_TAB} &
        PIDS+=($!)
    done
    echo "waiting for all tasks with sequence length ${MAX_SEQ_LENGTH}..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || { echo "error: process $pid failed!"; exit 1; }
    done
    PIDS=()  # 重置PID数组
    echo "all task with sequence length ${MAX_SEQ_LENGTH} are done."
done
