# Grid-Attention: Enhancing Computational Efficiency of Large Vision Models without Fine-Tuning

## Integrating GridAttn to SAM and Expedit-SAM
We fork the repos from the following and intergrate GridAttn on them:
-   Expedit-SAM: https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM.git 
(commit id: 75a1c30d66ad66999cb80fdfc85829f7e71dcacf)    
-   Detectron2: https://github.com/facebookresearch/detectron2    
(commit id: 7d2e68dbe452fc422268d40ac185ea2609affca8)

### Install envoriment
```
# Init env
conda create -n gridattn_sam python=3.8.3
conda activate gridattn_sam

# Install pytorch
pip install torch==1.12.1 torchvision==0.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install other packages 
# for calculate flops
pip install --upgrade git+https://github.com/llipengyu007/flops-counter.pytorch
# for LVIS dataset evaluation
pip install lvis

# Install Expedit-SAM
cd GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM
pip install -v -e .
cd ../../

# Install detectron2 for COCO and LVIS evaluation
cd GridAttn_on_SAM_and_ExpeditSAM/detectron2
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Link the SAM ckpts
```
# link SAM models
cd GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM
# To 'GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM' folder
ln -sf YOUR_PATH/sam_vit_b_01ec64.pth models/sam_vit_b.pth
ln -sf YOUR_PATH/sam_vit_l_0b3195.pth models/sam_vit_l.pth
ln -sf YOUR_PATH/sam_vit_h_4b8939.pth models/sam_vit_h.pth
cd ../../

# link VITDET models
cd GridAttn_on_SAM_and_ExpeditSAM/detectron2
ln -sf YOUR_PATH/cascade_mask_rcnn_vitdet_h/model_final_f05665.pkl models/coco_cascade_mask_rcnn_vitdet_h.pth
```

### Run a demo of expedit-sam, sam, and gridattn
```
cd GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM
sh demo.sh
# when grid_stride=1 means do not employ the GridAttn
cd ../../
```

### Calculate FLOPs for expedit-sam, sam, and gridattn
```
cd GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM
sh flops.sh
# when grid_stride=1 means do not employ the GridAttn
cd ../../
```

## Calculate latency for expedit-sam, sam, and gridattn
```
cd GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM
sh latency.sh
# when grid_stride=1 means do not employ the GridAttn
cd ../../
```


### Evaluate expedit-sam, sam, and gridattn on COCO and LVIS
```
# link vitdet models
cd GridAttn_on_SAM_and_ExpeditSAM/detectron2
#link coco model
ln -sf YOUR_PATH/cascade_mask_rcnn_vitdet_h/model_final_f05665.pkl models/coco_cascade_mask_rcnn_vitdet_h.pkl
#link lvis model
ln -sf YOUR_PATH/cascade_mask_rcnn_vitdet_h/model_final_11bbb7.pkl models/lvis_cascade_mask_rcnn_vitdet_h.pkl

# link coco and lvis datasets
ln -sf YOUR_PATH/COCO datasets/coco
ln -sf YOUR_PATH/LVIS datasets/lvis

# evaluate on coco and lvis datasets
sh eval_sam_coco.sh
sh eval_sam_lvis.sh
```
