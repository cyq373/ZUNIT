NAME='birds'
TIME_DIR=2021_02_23_21_39_06
EPOCH='200'
HOW_MANY=50
IMAGE_SIZE=128
ATTR_C_NUM=312

# training
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./dataset/${NAME} \
  --checkpoints_dir ./checkpoints \
  --time_dir ${TIME_DIR} \
  --name ${NAME} \
  --imageSize ${IMAGE_SIZE} \
  --how_many ${HOW_MANY} \
  --which_epoch ${EPOCH} \
  --attr_dim ${ATTR_C_NUM} \
  --is_flip 0 \
  --batchSize 1 \
  --seed 1
