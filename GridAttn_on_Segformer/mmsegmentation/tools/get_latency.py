# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import time

import numpy as np
import torch
import tqdm
from mmcv.parallel import scatter_kwargs
from torch.backends import cudnn

from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description='EasyCV model memory and inference_time test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--repeat_num', default=300, type=int, help='repeat number')
    parser.add_argument(
        '--warmup_num', default=100, type=int, help='warm up number')
    parser.add_argument(
        '--gpu',
        default='0',
        type=str,
        choices=['0', '1', '2', '3', '4', '5', '6', '7'])

    args = parser.parse_args()
    return args


def main():
    cudnn.benchmark = True

    args = parse_args()


    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    device = torch.device('cuda:{}'.format(args.gpu))
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model = model.to(device)
    model.eval()

    #cfg.data.val.samples_per_gpu = 1  # pop useless params
    #cfg.data.val.workers_per_gpu = 1  # pop useless params
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
    )

    # Set up cuda events for measuring time. This is PyTorch's official recommended interface and should theoretically be the most reliable.
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    # Initialize a time container.
    timings = np.zeros((args.repeat_num, 1))

    cnt_num = 0
    cost = 0
    with torch.no_grad():
        for data in data_loader:
            _, kwargs = scatter_kwargs(None, data, [int(args.gpu)])
            inputs = kwargs[0]
            inputs.update(dict(return_loss=False))
            for idx in tqdm.trange(args.repeat_num):
                # GPU may be hibernated to save energy at ordinary times, so it needs to be preheated.
                if idx < args.warmup_num:
                    if idx == 0:
                        print('Start warm up ...')
                    _ = model(**inputs)
                    continue

                if idx == args.warmup_num:
                    print('Warm up end, start to record time...')

                cost_1 = time.time()
                starter.record()
                _ = model(**inputs)
                ender.record()
                torch.cuda.synchronize()  # Wait for the GPU task to complete.
                curr_time = starter.elapsed_time(
                    ender)  # The time between starter and ender, in milliseconds.
                timings[idx] = curr_time
                cost += time.time() - cost_1
                cnt_num += 1
            break

    avg = timings.sum() / cnt_num
    cost = cost / cnt_num
    print('Cuda memory: {}'.format(torch.cuda.memory_summary(device)))
    print('\ninference average time={}ms\n'.format(avg))
    print('\ninference average time_2 ={}ms\n'.format(cost*1000))


if __name__ == '__main__':
    main()

