hourglass_num_cluster=144
device=cuda
gpu_id=0
grid_stride=1
model=b

log_root=./log/demo
mkdir -p ${log_root}


CUDA_VISIBLE_DEVICES="$gpu_id" python ./scripts/sam_demo.py \
  --device ${device} \
  --sam_checkpoint ./models/sam_vit_${model}.pth \
  --boxes 800 517 1786 987 \
  --input_image ./assets/demo2.jpg \
  --grid_stride ${grid_stride} \
  --hourglass_num_cluster $hourglass_num_cluster \
  --use_hourglass \
  --output_dir ${log_root}
