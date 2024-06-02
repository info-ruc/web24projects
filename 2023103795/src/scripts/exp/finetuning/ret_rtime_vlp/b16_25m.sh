export MASTER_PORT=$((11000 + $RANDOM % 20000))
export CUDA_VISIBLE_DEVICES=2,4,6,7
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='webvid_without_rewrite_1pos1neg_novtm'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=4
NUM_CPU=112

# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     -n${NNODE} \
#     --gres=gpu:${NUM_GPUS} \
#     --ntasks-per-node=1 \
#     --cpus-per-task=${NUM_CPU} \

    torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    tasks/retrieval_webvid.py \
    $(dirname $0)/b16_25m.py \
    pretrained_path /datassd2/pretrained_models/Unmasked_Teacher/multimodality/b16_25m.pth \
    output_dir ${OUTPUT_DIR} \
