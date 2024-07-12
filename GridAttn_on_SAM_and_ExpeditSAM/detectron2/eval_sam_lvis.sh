hourglass_num_cluster=144
gpu_id=1
log_root='./log/LVIS'
mkdir -p ${log_root}

for grid_stride in 1 2
do
for model in b l h
do
CUDA_VISIBLE_DEVICES="$gpu_id" python ./tools/lazyconfig_train_net.py \
  --hourglass_num_cluster ${hourglass_num_cluster} \
  --grid_stride  ${grid_stride} \
  --sam_checkpoint ../Expedit-SAM/models/sam_vit_${model}.pth \
  --config-file ./projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py \
  --eval-only \
  --use_hourglass \
  train.init_checkpoint=./models/lvis_cascade_mask_rcnn_vitdet_h.pkl \
  train.output_dir="${log_root}/expedsam_${hourglass_num_cluster}_${model}_${grid_stride}_cuda"

CUDA_VISIBLE_DEVICES="$gpu_id" python ./tools/lazyconfig_train_net.py \
  --hourglass_num_cluster ${hourglass_num_cluster} \
  --grid_stride  ${grid_stride} \
  --sam_checkpoint ../Expedit-SAM/models/sam_vit_${model}.pth \
  --config-file ./projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py \
  --eval-only \
  train.init_checkpoint=./models/lvis_cascade_mask_rcnn_vitdet_h.pkl \
  train.output_dir="${log_root}/sam_${model}_${grid_stride}_cuda"

done
done