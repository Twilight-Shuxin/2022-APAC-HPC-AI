#!/bin/bash

CFG_FILE=./cluster.cfg
if [ ! -f $CFG_FILE ]; then
    echo "Config file ${CFG_FILE} not found. Make sure you're submitting the "
    echo "job from the directory containing launch.sh and cluster.cfg files."
fi

. ${CFG_FILE}

SERVER_PROC=$1
NODE_NUM=$2

if [ "$SERVER_PROC" = true ]; then
    LOG_FILE=${JOB_DIR}/server-${HOSTNAME}-${JOBID}.log
    OUT_FILE=${JOB_DIR}/server-${HOSTNAME}-${JOBID}.out
else
    LOG_FILE=${JOB_DIR}/worker-${HOSTNAME}-${JOBID}.log
    OUT_FILE=${JOB_DIR}/worker-${HOSTNAME}-${JOBID}.out
fi

SERVER_FILE=${JOB_DIR}/.server-${JOBID}.json

cd $TEST_DIR

echo "hostname: $(hostname)" >> ${LOG_FILE}
echo "PWD: ${PWD}" >> ${LOG_FILE}

nvidia-smi >> ${LOG_FILE}
nvidia-smi topo -m >> ${LOG_FILE}

# Setup conda
source ${WORK_DIR}/miniconda/etc/profile.d/conda.sh
conda activate /data/home/apac2022/miniconda3/envs/ucx
echo "CONDA_PREFIX: ${CONDA_PREFIX}" >> ${LOG_FILE}

echo "JOBID: ${JOBID}" >> ${LOG_FILE}
echo "NGPUS: ${NGPUS}" >> ${LOG_FILE}
echo "WORK_DIR: ${WORK_DIR}" >> ${LOG_FILE}

echo "SERVER_PROC: ${SERVER_PROC}" >> ${LOG_FILE}
echo "SERVER_FILE: ${SERVER_FILE}" >> ${LOG_FILE}
echo "NUM_ITERATIONS: ${NUM_ITERATIONS}" >> ${LOG_FILE}
echo "CHUNK_SIZE: ${CHUNK_SIZE}" >> ${LOG_FILE}
echo "NODE_NUM: ${NODE_NUM}" >> ${LOG_FILE}

if [ "$SERVER_PROC" = true ]; then
    pwd
    echo "$ python -m cProfile -o server.prof cudf-merge-test1.py  \
        --server \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${NGPUS} \
        --iter ${NUM_ITERATIONS} \
        -c ${CHUNK_SIZE} " &>> ${OUT_FILE}
    # python -m cProfile -o server.prof cudf-merge.py \
    # vtune -collect io \
    # python cudf-merge-test0.py \
    python cudf-merge-test2.py \
        --server \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${NGPUS} \
        --iter ${NUM_ITERATIONS} \
        -c ${CHUNK_SIZE}  >> ${OUT_FILE}
    # python data_process3.py -l="350" -d=4 -t=10 >>prof.out
    #python workflow_process2.py -k=4 -t=10 -d=4
    # Clean server file
    rm -f ${SERVER_FILE}

else
    # Wait for server to come up
    while [ ! -f ${SERVER_FILE} ]; do
        sleep 3
    done
    sleep 3

    if [ "$WRITE_BASELINE" = true ]; then
        RESULTS_ARGS="--write-results-to-disk ${BASELINE_DIR}"
    fi
    if [ "$VERIFY_RESULTS" = true ]; then
        RESULTS_ARGS="--verify-results ${BASELINE_DIR}"
    fi


    echo "$ python -m cProfile -o worker.prof cudf-merge-test1.py  \
        --devs "${DEVICES}" \
        --rmm-init-pool-size 30000000000 \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${NGPUS} \
        --node-num ${NODE_NUM} \
        --iter ${NUM_ITERATIONS} \
        --start-rank ${START} \
        -c ${CHUNK_SIZE} \
        ${RESULTS_ARGS} " &>> ${OUT_FILE} &
    # python -m cProfile -o worker-${HOSTNAME}.prof cudf-merge.py  \
    # python cudf-merge-test0.py \
    python cudf-merge-test2.py \
        --devs "${DEVICES}" \
        --rmm-init-pool-size 30000000000 \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${NGPUS} \
        --node-num ${NODE_NUM} \
        --iter ${NUM_ITERATIONS} \
        --start-rank ${START} \
        -c ${CHUNK_SIZE} \
        ${RESULTS_ARGS}  &>> ${OUT_FILE} &
fi

echo "$HOSTNAME completed at: $(date)" >> ${LOG_FILE}
