mkdir ./latency

repeat=200
device=cpu
gpu_id=7


#for grid_stride in 1 2 4 8
#do
#for model in b l h
#do
#
#CUDA_VISIBLE_DEVICES="$gpu_id" python grounded_sam_demo.py \
#  --device ${device} \
#  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#  --grounded_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/grounding_dino/groundingdino_swint_ogc.pth \
#  --sam_hq_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_hq_vit_${model}.pth \
#  --sam_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_vit_${model}.pth \
#  --output_dir outputs/gridattn_strid_tmp \
#  --box_threshold 0.3 --text_threshold 0.25 \
#  --input_image /home/Rhossolas.Lee/code/IDEA/Grounded-Segment-Anything/assets/demo2.jpg \
#  --grid_stride ${grid_stride} \
#  --repeat_times ${repeat} \
#  --text_prompt "dog" \
#  > ./latency/sam_${model}_${device}_${grid_stride}.txt
#
#done
#done



export PYTHONPATH="/home/Rhossolas.Lee/code/Expedit-SAM:$PYTHONPATH"

for grid_stride in 1 2 4 8
do
for model in b l h
do

CUDA_VISIBLE_DEVICES="$gpu_id" python grounded_sam_demo.py \
  --device ${device} \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/grounding_dino/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_hq_vit_${model}.pth \
  --sam_checkpoint /home/Rhossolas.Lee/.cache/groundSAM/segment_anything/sam_vit_${model}.pth \
  --output_dir outputs/gridattn_strid_tmp \
  --box_threshold 0.3 --text_threshold 0.25 \
  --input_image /home/Rhossolas.Lee/code/IDEA/Grounded-Segment-Anything/assets/demo2.jpg \
  --grid_stride ${grid_stride} \
  --repeat_times ${repeat} \
  --text_prompt "dog" \
  --hourglass_num_cluster 256 \
  > ./latency/expedsam_${model}_${device}_${grid_stride}.txt

done
done