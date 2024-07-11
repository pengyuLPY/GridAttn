cluster=144
log_root=./log/flops

mkdir -p ${log_root}

for grid_stride in 1 2 4 8
do
for model in b l h
do

python ./scripts/sam_flops.py \
  --device cuda \
  --grid_stride ${grid_stride} \
  --hourglass_num_cluster ${cluster} \
  --sam_checkpoint vit_${model} \
  --use_hourglass \
  > ${log_root}/expedsam_${cluster}_${model}_${grid_stride}_cuda.txt

python ./scripts/sam_flops.py \
  --device cuda \
  --grid_stride ${grid_stride} \
  --hourglass_num_cluster ${cluster} \
  --sam_checkpoint vit_${model} \
  > ${log_root}/sam_${model}_${grid_stride}_cuda.txt

done
done