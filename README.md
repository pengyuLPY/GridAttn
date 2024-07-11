coming soon
# integrating GridAttn On SAM and Expedit-SAM


## install envoriment
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

## calculate FLOPs for expedit-sam, sam, and gridattn
```
# in 'GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM' folder
sh flops.sh
# when grid_stride=1 means do not employ the GridAttn
```


