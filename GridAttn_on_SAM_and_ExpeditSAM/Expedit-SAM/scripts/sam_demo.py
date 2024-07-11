import argparse
import os
import torch
from PIL import Image

# segment anything
from segment_anything import (
    build_sam_vit_b,
    build_sam_vit_l,
    build_sam_vit_h,
    SamPredictor
)


import segment_anything
print("SAM PATH:{}".format(segment_anything.__path__))


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def save_mask_data(output_dir, mask_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("GridAttn with SAM/Expedit-SAM Latency", add_help=True)

    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--boxes", type=int, nargs="+", help="box")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--grid_stride", type=int, default=1, help="using str type, 1 means no use, positive value only use in global attn, negative vaule means use all attn" )
    parser.add_argument("--repeat_times", type=int, default=1, help="repeat times for time costs")

    parser.add_argument("--use_hourglass", action="store_true", help="using Expedit-SAM for prediction")
    parser.add_argument("--hourglass_num_cluster", type=int, default=144)

    args = parser.parse_args()

    # cfg
    sam_checkpoint = args.sam_checkpoint
    grid_stride = args.grid_stride
    use_hourglass = args.use_hourglass
    image_path = args.input_image
    output_dir = args.output_dir
    boxes = np.array([args.boxes])
    device = args.device
    repeat_times = args.repeat_times
    use_hourglass = args.use_hourglass
    hourglass_num_cluster = args.hourglass_num_cluster


    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil = Image.open(image_path).convert("RGB")

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize SAM
    if sam_checkpoint.find("vit_h") >= 0:
        build_sam = build_sam_vit_h
    elif sam_checkpoint.find("vit_l") >= 0:
        build_sam = build_sam_vit_l
    elif sam_checkpoint.find("vit_b") >= 0:
        build_sam = build_sam_vit_b
    else:
        assert False, sam_checkpoint[-6:]


    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint,
                                       grid_stride=grid_stride,
                                       hourglass_num_cluster=hourglass_num_cluster,
                                       use_hourglass=use_hourglass,
                                       ).to(device))


    cost_set_image = 0
    for tt in tqdm(range(repeat_times)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.reset_image()
        start = time.time()
        predictor.set_image(image)
        cost_set_image += time.time() - start

        size = image_pil.size
        H, W = size[1], size[0]

        boxes_filt = torch.Tensor(boxes)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )


        if tt < 1:
            # draw output image
            start = time.time()
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, "grounded_sam_output.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )

            save_mask_data(output_dir, masks)
            plt.cla()
            plt.clf()
            plt.close('all')


    print("cost_set_image={:.4f}".format(cost_set_image/repeat_times))
    print("complete", repeat_times, grid_stride)

