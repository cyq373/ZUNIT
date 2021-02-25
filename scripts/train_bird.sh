NAME='birds'
IMAGE_SIZE=128
ATTR_C_NUM=312
IS_FLIP=1
NITER=100
NITER_DECAY=100
SAVE_EPOCH=50
BATCHSIZE=8

# training
GPU_ID=0
PORT=$((GPU_ID+8080)) 
DISPLAY_ID=$((GPU_ID*347+3))
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataroot  ./dataset/${NAME} \
  --checkpoints_dir ./checkpoints \
  --display_id ${DISPLAY_ID} \
  --name ${NAME} \
  --imageSize ${IMAGE_SIZE} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --attr_dim ${ATTR_C_NUM} \
  --seen_classes_num 150 \
  --save_epoch_freq ${SAVE_EPOCH} \
  --is_flip ${IS_FLIP} \
  --display_port ${PORT} \
  --batchSize ${BATCHSIZE} \
  --init_type kaiming \
  --display_freq 1600
