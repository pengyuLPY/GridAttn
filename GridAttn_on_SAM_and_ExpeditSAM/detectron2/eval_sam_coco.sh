hourglass_num_cluster=144
gpu_id=2
log_root='./log/COCO'
mkdir -p ${log_root}

for grid_stride in 1 2
do
for model in b l h
do
CUDA_VISIBLE_DEVICES="$gpu_id" python ./tools/lazyconfig_train_net.py \
  --hourglass_num_cluster ${hourglass_num_cluster} \
  --grid_stride  ${grid_stride} \
  --sam_checkpoint ../Expedit-SAM/models/sam_vit_${model}.pth \
  --config-file ./projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py \
  --eval-only \
  --use_hourglass \
  train.init_checkpoint=./models/coco_cascade_mask_rcnn_vitdet_h.pkl \
  train.output_dir="${log_root}/expedsam_${hourglass_num_cluster}_${model}_${grid_stride}_cuda"

CUDA_VISIBLE_DEVICES="$gpu_id" python ./tools/lazyconfig_train_net.py \
  --hourglass_num_cluster ${hourglass_num_cluster} \
  --grid_stride  ${grid_stride} \
  --sam_checkpoint ../Expedit-SAM/models/sam_vit_${model}.pth \
  --config-file ./projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py \
  --eval-only \
  train.init_checkpoint=./models/coco_cascade_mask_rcnn_vitdet_h.pkl \
  train.output_dir="${log_root}/sam_${model}_${grid_stride}_cuda"

done
done

#CUDA_VISIBLE_DEVICES="0" python ./tools/lazyconfig_train_net.py \
#  --hourglass_num_cluster 256 \
#  --grid_stride 2 \
#  --sam_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_vit_b.pth \
#  --config-file ./projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py \
#  --eval-only train.init_checkpoint=/home/Rhossolas.Lee/.cache/groundSAM/vitdet/cascade_mask_rcnn_vitdet_h/LVIS/model_final_11bbb7.pkl \
#  train.output_dir="./output/Expedit_SAM_LVIS_256/vit_b_gs_2"

