#!/bin/bash
ENV="${HOME}/anaconda3/envs/hrnetnc/bin/python3"
CODE_DIR="../"
DATA_DIR="/data/datasets/MegaDepth_v1_SfM/"
TRAINEXE="${CODE_DIR}/train.py"
DATASET_TRAINING_FILE="${DATA_DIR}/training_pairs.txt"
DATASET_VALIDATION_FILE="${DATA_DIR}/validation_pairs.txt"
DATASET_IMAGE_LOCATION="${DATA_DIR}"
CONFIG="xrcnet.json"
CHECKPOINT=""
GPU="0"

currentcommand="${0}"
currentargs="${@}"

usage="$(basename "$0") [-h] [-c ${CONFIG}] [-m ${CHECKPOINT}] [-g ${GPU}]
Runs training of xrcnet

where:
    -h                                   shows this help text
    -c  [<path/to/configuration.json>]   configuration file specifying the different training params
    -m  [<path/to/checkpoint/model>]     checkpoint model to load weights from (LR WILL BE OVERWRITTEN)
    -g  [<list of int>]                  which GPU(s) to run it on"

while getopts ':h:c:m:g:' option; do
  case "$option" in
    h)
      echo "$usage"
      exit
      ;;
    c)
      CONFIG=$(realpath ${OPTARG})
      ;;
    m)
      CHECKPOINT=$(realpath ${OPTARG})
      ;;
    g)
      GPU=${OPTARG}
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


MODELS_DIR="${CODE_DIR}/trained_models/"

if [ ! -f ${CONFIG} ]; then
    echo "Please specify config json file!"
    exit 1
fi

# complain if either the code or the data is missing
if [ ! -d "${CODE_DIR}" ]
then
  echo "${CODE_DIR} does not exist. Please check the correct path to the code."
  exit 2
fi

if [ ! -d "${DATA_DIR}" ]
then
  echo "${DATA_DIR} does not exist. Please check the correct path to HPatches."
  exit 3
fi



if [ ! -d "$MODELS_DIR" ]
then
  mkdir -p "$MODELS_DIR"
fi

CONFIG_BASENAME=$(basename -- ${CONFIG})
STDERRLOG="${MODELS_DIR}/xrcnet_${CONFIG_BASENAME::-5}.txt"

if [ -z "${CHECKPOINT}" ]
then
  echo "CHECKPOINT NOT PROVIDED. Will train from scratch."
  CUDA_VISIBLE_DEVICES=${GPU} ${ENV} ${TRAINEXE} --training_file ${DATASET_TRAINING_FILE} --validation_file ${DATASET_VALIDATION_FILE} --image_path ${DATASET_IMAGE_LOCATION} --config ${CONFIG} --result_model_dir ${MODELS_DIR} > ${STDERRLOG} 2>&1 &
else
  echo "Loading checkpoint from ${CHECKPOINT}. Will load the weights from there."
  CUDA_VISIBLE_DEVICES=${GPU} ${ENV} ${TRAINEXE} --checkpoint ${CHECKPOINT} --training_file ${DATASET_TRAINING_FILE} --validation_file ${DATASET_VALIDATION_FILE} --image_path ${DATASET_IMAGE_LOCATION} --config ${CONFIG} --result_model_dir ${MODELS_DIR} > ${STDERRLOG} 2>&1 &
fi
