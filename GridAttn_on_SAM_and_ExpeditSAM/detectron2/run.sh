#CUDA_VISIBLE_DEVICES="7" python ./tools/lazyconfig_train_net.py \
#  --hourglass_num_cluster 256 \
#  --grid_stride  8 \
#  --sam_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_vit_b.pth \
#  --config-file ./projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py \
#  --eval-only train.init_checkpoint=/home/Rhossolas.Lee/.cache/groundSAM/vitdet/cascade_mask_rcnn_vitdet_h/model_final_f05665.pkl \
#  train.output_dir="./output/Expedit_SAM_COCO_256/use_hourglass_vit_b_gs_8"


CUDA_VISIBLE_DEVICES="2" python ./tools/lazyconfig_train_net.py \
  --hourglass_num_cluster 256 \
  --grid_stride 2 \
  --sam_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_vit_b.pth \
  --config-file ./projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py \
  --eval-only train.init_checkpoint=/home/Rhossolas.Lee/.cache/groundSAM/vitdet/cascade_mask_rcnn_vitdet_h/LVIS/model_final_11bbb7.pkl \
  train.output_dir="./output/Expedit_SAM_LVIS_256/vit_b_gs_2"

