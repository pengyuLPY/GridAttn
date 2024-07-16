# Grid-Attention: Enhancing Computational Efficiency of Large Vision Models without Fine-Tuning

## Integrating GridAttn to SAM and Expedit-SAM
We fork the repos from the following and intergrate GridAttn on them:    
-   Expedit-SAM: https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM.git     
(Based commit id: 75a1c30d66ad66999cb80fdfc85829f7e71dcacf, main branch)    
-   Detectron2: https://github.com/facebookresearch/detectron2        
(Based commit id: 7d2e68dbe452fc422268d40ac185ea2609affca8, main branch)

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
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ../../

# Install detectron2 for COCO and LVIS evaluation
cd GridAttn_on_SAM_and_ExpeditSAM/detectron2
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Link the models
```
# link SAM models
cd GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM
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


## Integrating GridAttn to SD
We fork the repos from the following and intergrate GridAttn on them:
-   diffusers: https://github.com/huggingface/diffusers.git    
(Based commit id: 3045fb276352681f6b9075956e599dd8ef571872, main branch)

### Install envoriment
```
# Init env
conda create -n gridattn_sd python=3.8.3
conda activate gridattn_sd

# Install pytorch
pip install torch==2.0.0 torchvision==0.15.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install other packages 
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple


# Install diffusers
cd GridAttn_on_SD/diffusers
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ../../
```

### Link the models
```
# link SD models
cd GridAttn_on_SD/scripts
ln -sf YOUR_PATH/stable-diffusion-2-1 models/stable-diffusion-2-1-gridattn

#copy GridAttn config to unet folder
cp models/unet_config_gridattn.json models/stable-diffusion-2-1/unet/config.json
```

### Run a demo of SD with gridattn
```
cd GridAttn_on_SD/scripts
python demo.py
# when grid_stride=1 means do not employ the GridAttn
cd ../../
```


## Integrating GridAttn to Segformer
We fork the repos from the following and intergrate GridAttn on them:
-   mmsegmentation: https://github.com/open-mmlab/mmsegmentation    
(Based commit id: ca7c098767371e633f5672d128e5808dd9fb7634, master branch)

### Install envoriment
```
# Init env
conda create -n gridattn_seg python=3.8.3
conda activate gridattn_seg

# Install pytorch
pip install torch==1.12.1 torchvision==0.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple


# Install other packages 
pip install openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmcv-full==1.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install mmsegmentation
cd GridAttn_on_Segformer/mmsegmentation
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ../../
```


### Link the models
```
cd GridAttn_on_SD
ln -sf YOUR_PATH ./models/
cd ../
```


## Calculate latency for Segformer with Gridattn
```
cd GridAttn_on_Segformer/scripts/latency
sh latency.sh
cd ../../../
```

### Evaluate Segformer with Gridattn on ADE20K
```
cd GridAttn_on_Segformer/scripts/eval

# link the dataset
ln -sf YOUR_ADE20K_PATH ./data

# run the script
sh test.sh

cd ../../../
```


### Train Segformer with Gridattn on ADE20K
```
cd GridAttn_on_Segformer/scripts/train_from_scratch

# link the dataset
ln -sf YOUR_ADE20K_PATH ./data

# run the script
sh train.sh

cd ../../../
```


## Citation
```latex
@article{liang2022expediting,
  title={Expediting large-scale vision transformer for dense prediction without fine-tuning},
  author={Liang, Weicong and Yuan, Yuhui and Ding, Henghui and Luo, Xiao and Lin, Weihong and Jia, Ding and Zhang, Zheng and Zhang, Chao and Hu, Han},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={35462--35477},
  year={2022}
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10684--10695},
  year={2022}
}

@article{xie2021segformer,
  title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={12077--12090},
  year={2021}
}
```