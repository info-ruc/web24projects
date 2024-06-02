export MASTER_PORT=$((11000 + $RANDOM % 20000))
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='b16_25m_100k_12frame_1k'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=1
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
    tasks/retrieval.py \
    $(dirname $0)/test.py \
    pretrained_path /data2/dy/code/unmasked_teacher/multi_modality/exp/finetuning/ret_rtime/b16_25m_100k_12frame/ckpt_best.pth \
    output_dir ${OUTPUT_DIR} \
