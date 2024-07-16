for i in `seq 0 5`
do

#(
 CONFIG=./segformer_b${i}_init.py
 OUT=./result_scratch_segformer_b${i}.pkl
 CHECKPOINT=../../models/ADE20K/train_from_scratch_gridattn_gs2/segformer_b${i}_init.pth
 
 GPUS=$i
 NNODES=${NNODES:-1}
 NODE_RANK=${NODE_RANK:-0}
 PORT=29590
 MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
 MMSEG=../../mmsegmentation
 
 
 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 CUDA_VISIBLE_DEVICES="$GPUS" python $MMSEG/tools//test.py \
     $CONFIG \
     $CHECKPOINT \
    --eval mIoU\
    --out result.pkl \
    --gpu-id 0 
 
 
 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 CUDA_VISIBLE_DEVICES="$GPUS" python $MMSEG/tools//test.py \
     $CONFIG \
     $CHECKPOINT \
    --eval mIoU\
    --out result.pkl \
    --gpu-id 0 \
    --aug-test
#) &

done
