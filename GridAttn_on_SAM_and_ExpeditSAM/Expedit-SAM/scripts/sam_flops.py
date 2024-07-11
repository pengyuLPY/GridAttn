import argparse
from ptflops import get_model_complexity_info

# segment anything
from segment_anything import (
    SamPredictor,
    build_sam_vit_b,
    build_sam_vit_l,
    build_sam_vit_h,
)

import segment_anything
print("SAM PATH:{}".format(segment_anything.__path__))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("GridAttn with SAM/Expedit-SAM FLOPs", add_help=True)
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--grid_stride", type=int, default=1,
                        help="using str type, 1 means no use, positive value only use in global attn, negative vaule means use all attn")
    parser.add_argument("--hourglass_num_cluster", type=int, default=144)
    parser.add_argument("--use_hourglass", action="store_true", help="using Expedit-SAM for prediction")
    args = parser.parse_args()

    # cfg

    sam_checkpoint = args.sam_checkpoint
    grid_stride = args.grid_stride
    use_hourglass = args.use_hourglass

    image_size = (3, 1024, 1024)

    device = args.device
    hourglass_num_cluster = args.hourglass_num_cluster


    # initialize SAM
    if sam_checkpoint.find("vit_h") >= 0:
        predictor = SamPredictor(build_sam_vit_h(grid_stride=grid_stride,
                                                 hourglass_num_cluster=hourglass_num_cluster,
                                                 use_hourglass=use_hourglass).to(device))
    elif sam_checkpoint.find("vit_l") >= 0:
        predictor = SamPredictor(build_sam_vit_l(grid_stride=grid_stride,
                                                 use_hourglass=use_hourglass,
                                                 hourglass_num_cluster=hourglass_num_cluster).to(device))
    elif sam_checkpoint.find("vit_b") >= 0:
        predictor = SamPredictor(build_sam_vit_b(grid_stride=grid_stride,
                                                 use_hourglass=use_hourglass,
                                                 hourglass_num_cluster=hourglass_num_cluster).to(device))
    else:
        assert False, "sam_checkpoint error {}".format(sam_checkpoint)

    macs, params = get_model_complexity_info(predictor.model.image_encoder, image_size, as_strings=True,
                                       print_per_layer_stat=True, verbose=True)

    print("grid_stride:", grid_stride)
    print("macs", macs)
    print("params", params)
    print("complete")

