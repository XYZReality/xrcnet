#!/bin/bash
ENV="${HOME}/anaconda3/envs/hrnetnc/bin/python3"
CODE_DIR="../"
DATA_DIR="/data/datasets/HPatches/"
EVALEXE="${CODE_DIR}/HPatches/eval_hpatches.py"
SEQUENCE_LIST="${CODE_DIR}/HPatches/image_list_hpatches_sequences.txt"
OUTPUT_DIR="${CODE_DIR}/HPatches/evaluations/"
CHECKPOINT=""
GPU="0"
EXPERIMENT_NAME="xrcnet"
IMAGESIZE="3000"

currentcommand="${0}"
currentargs="${@}"

usage="$(basename "$0") [-h] [-e ${EXPERIMENT_NAME}] [-m ${CHECKPOINT}] [-g ${GPU}] [-s ${IMAGESIZE}]
Runs model evaluation on HPatches

where:
    -h                                   shows this help text
    -e  [<string>]   			         name of this experiment
    -m  [<path/to/checkpoint/model>]     checkpoint model to load weights from
    -g  [<list of int>]                  which GPU(s) to run it on
    -s  [int]                            crop image to this size (defaults to 1600)
    "

while getopts ':h:e:m:g:f:b:s:c:k:n:' option; do
  case "$option" in
    h)
      echo "$usage"
      exit
      ;;
    e)
      EXPERIMENT_NAME=${OPTARG}
      ;;
    m)
      CHECKPOINT=$(realpath ${OPTARG})
      ;;
    g)
      GPU=${OPTARG}
      ;;
    s)
      IMAGESIZE=${OPTARG}
      ;;
    :)
      printf "missing argument for -%s\n" "$OPTARG" >&2
      echo "$usage" >&2
      exit 1
      ;;
   \?)
      printf "illegal option: -%s\n" "$OPTARG" >&2
      echo "$usage" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [ ! -f ${CONFIG} ]; then
    echo "Please specify config json file!"
    exit 2
fi

# complain if either the code or the data is missing
if [ ! -d "${CODE_DIR}" ]
then
  echo "${CODE_DIR} does not exist. Please check the correct path to the code."
  exit 3
fi

if [ ! -d "${DATA_DIR}" ]
then
  echo "${DATA_DIR} does not exist. Please check the correct path to HPatches."
  exit 4
fi

# detect if it's multi-gpu evaluation
GPUARR=(${GPU//,/ })
NUMGPUS=${#GPUARR[@]}
MULTIGPU=""

if [ ${NUMGPUS} -gt 1 ];
then
  MULTIGPU="True"
fi

# create a directory to store stderr
STDERRDIR="${OUTPUT_DIR}/processed_logs"

if [ ! -d "${STDERRDIR}" ]
then
  mkdir -p ${STDERRDIR}
fi

STDERRLOG="${STDERRDIR}/${EXPERIMENT_NAME}_output.txt"

if [ -z "${CHECKPOINT}" ]
then
  echo "Please provide a checkpoint to evaluate."
else
    if [ -z "${MULTIGPU}" ]
    then
        echo "Loading checkpoint from ${CHECKPOINT}. Starating evaluation on GPU ${GPU}."
        CUDA_VISIBLE_DEVICES=${GPU} ${ENV} ${EVALEXE} --checkpoint ${CHECKPOINT} --root ${DATA_DIR} --sequence_list ${SEQUENCE_LIST} --output_dir ${OUTPUT_DIR} --experiment_name ${EXPERIMENT_NAME} --image_size ${IMAGESIZE} > ${STDERRLOG} 2>&1 &
    else
        echo "Loading checkpoint from ${CHECKPOINT}. Starating evaluation on multiple GPUs: ${GPU}."
        CUDA_VISIBLE_DEVICES=${GPU} ${ENV} ${EVALEXE} --checkpoint ${CHECKPOINT} --root ${DATA_DIR} --sequence_list ${SEQUENCE_LIST} --output_dir ${OUTPUT_DIR} --experiment_name ${EXPERIMENT_NAME} --image_size ${IMAGESIZE} --multi_gpu_eval ${MULTIGPU} > ${STDERRLOG} 2>&1 &
    fi
fi
        


