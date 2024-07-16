ROOT=./log
mkdir $ROOT


for i in `seq 0 5`
do

CONFIG=./segformer_b${i}_init.py
MMSEG=../../mmsegmentation
DIR=$ROOT/segformer_b${i}_init.log


python $MMSEG/tools/get_latency.py \
    $CONFIG > $DIR \
    --repeat_num 1000 \
    --gpu "0"

done
