#!/bin/bash
CFG_FILE=./cluster.cfg
if [ ! -f $CFG_FILE ]; then
    echo "Config file ${CFG_FILE} not found. Make sure you're submitting the "
    echo "job from the directory containing launch.sh and cluster.cfg files."
fi

. ${CFG_FILE}

################################################################################
#         Launch server and workers -- shouldn't be necessary to change        #
################################################################################
# Load CUDA module
# module load cuda/11.4.1


JOB_ID=$1
echo $JOB_ID $APAC_DIR

make_dir() {
   DIRECTORY_NAME=$1
   echo $DIRECTORY_NAME
   if [ -z ${DIRECTORY_NAME} ]; then
       echo "Empty directory name, this probably means JOB_DIR is undefined"
       exit 1
   fi

   if [ ! -d ${DIRECTORY_NAME} ]; then
       mkdir -p ${DIRECTORY_NAME}
   fi
}

make_dir $JOB_DIR

echo "PWD: $PWD"
echo "SCHEDULER_HOST: ${SCHEDULER_HOST}"
echo "ALL_HOSTS: ${ALL_HOSTS}"

cd $APAC_DIR

echo "hostname: $(hostname)"
nvidia-smi
nvidia-smi topo -m

# Setup conda
source ${WORK_DIR}/miniconda/etc/profile.d/conda.sh
conda activate /data/home/apac2022/miniconda3/envs/ucx
echo $CONDA_PREFIX

python -c "import ucp; print(ucp.get_ucx_version())"
ucx_info -v

# Launch workers
NODE_NUM=0
RANK_CNT=0
for host in ${ALL_HOSTS[@]}; do
    echo ${host}
    echo "ssh -o StrictHostKeyChecking=no -n -f $host \
        \"cd $APAC_DIR; \
        TEST_DIR=${APAC_DIR} START=${RANK_CNT} JOBID=${JOB_ID} NGPUS=${NGPUS} WORKDIR=${WORK_DIR} DEVICES=${VISIBLE_DEVICES[$NODE_NUM]} nohup bash $APAC_DIR/run-cluster.sh \
        false \
        ${NODE_NUM} &> /dev/null &\""
    ssh -o StrictHostKeyChecking=no -n -f $host \
        "cd $APAC_DIR; \
        TEST_DIR=${APAC_DIR} START=${RANK_CNT} JOBID=${JOB_ID} NGPUS=${NGPUS} WORKDIR=${WORK_DIR} DEVICES=${VISIBLE_DEVICES[$NODE_NUM]} nohup bash $APAC_DIR/run-cluster.sh \
        false \
        ${NODE_NUM} &> /dev/null &"
        
    RANK_CNT=$(( ${RANK_CNT} + ${DEVICE_NUM[$NODE_NUM]} ))
    NODE_NUM=$(expr ${NODE_NUM} + 1)
done

# Launch server
echo "JOBID=${JOB_ID} TEST_DIR=${APAC_DIR} bash $APAC_DIR/run-cluster.sh true &> /dev/null"
JOBID=${JOB_ID} TEST_DIR=${APAC_DIR} bash $APAC_DIR/run-cluster.sh true

echo "Completed at $(date)"
