# Grid-Attention: Enhancing Computational Efficiency of Large Vision Models without Fine-Tuning 00

## Integrating GridAttn On SAM and Expedit-SAM
fork from https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM.git

### Install envoriment
```
# init env
conda create -n gridattn_sam python=3.8.3
conda activate gridattn_sam

# install pytorch
pip install torch==1.12.1 torchvision==0.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# install other package for calculate flops
# pip install --upgrade git+https://github.com/llipengyu007/flops-counter.pytorch

# install Expedit-SAM
cd GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM
pip install -v -e .
```

### Link the sam ckpt models
```
# To 'GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM' folder
ln -sf YOUR_PATH/sam_vit_b_01ec64.pth models/sam_vit_b.pth
ln -sf YOUR_PATH/sam_vit_l_0b3195.pth models/sam_vit_l.pth
ln -sf YOUR_PATH/sam_vit_h_4b8939.pth models/sam_vit_h.pth
```

### Run a demo of expedit-sam, sam, and gridattn
```
# To 'GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM' folder
sh demo.sh
# when grid_stride=1 means do not employ the GridAttn
```

### Calculate FLOPs for expedit-sam, sam, and gridattn
```
# To 'GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM' folder
sh flops.sh
# when grid_stride=1 means do not employ the GridAttn
```

## Calculate latency for expedit-sam, sam, and gridattn
```
# To 'GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM' folder
sh latency.sh
# when grid_stride=1 means do not employ the GridAttn
```




