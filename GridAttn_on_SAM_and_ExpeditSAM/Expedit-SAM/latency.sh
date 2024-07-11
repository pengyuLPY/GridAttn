hourglass_num_cluster=144
repeat=50
device=cuda
gpu_id=0

log_root=./log/latency
mkdir -p ${log_root}

for grid_stride in 1 2 4 8
do
for model in b l h
do

CUDA_VISIBLE_DEVICES="$gpu_id" python ./scripts/sam_demo.py \
  --device ${device} \
  --sam_checkpoint ./models/sam_vit_${model}.pth \
  --boxes 800 517 1786 987 \
  --input_image ./assets/demo2.jpg \
  --grid_stride ${grid_stride} \
  --repeat_times ${repeat} \
  --hourglass_num_cluster $hourglass_num_cluster \
  --use_hourglass \
  --output_dir ${log_root} \
  > ${log_root}/expedsam_${hourglass_num_cluster}_${model}_${grid_stride}_cuda.txt

CUDA_VISIBLE_DEVICES="$gpu_id" python ./scripts/sam_demo.py \
  --device ${device} \
  --sam_checkpoint ./models/sam_vit_${model}.pth \
  --boxes 800 517 1786 987 \
  --input_image ./assets/demo2.jpg \
  --grid_stride ${grid_stride} \
  --repeat_times ${repeat} \
  --hourglass_num_cluster $hourglass_num_cluster \
  --output_dir ${log_root} \
  > ${log_root}/sam_${model}_${grid_stride}_cuda.txt

done
done