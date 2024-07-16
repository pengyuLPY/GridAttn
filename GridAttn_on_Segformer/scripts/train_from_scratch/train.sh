for i in `seq 0 5`
do


CONFIG=./segformer_b${i}_init.py
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=29501
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MMSEG=../../mmsegmentation

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $MMSEG/tools//train.py \
    $CONFIG \
    --launcher pytorch ${@:3}


done
