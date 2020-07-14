import argparse

import numpy as np
import torch
import torch.nn as nn

from reconstruct_surface import MLP, upsample_surface
import point_cloud_utils as pcu


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("state_file", type=str, help="Path to a reconstructed surface state file generated with "
                                                        "`reconstruct_surface.py` or `reconstruct_single_patch.py`")
    argparser.add_argument("--devices", type=str, default="", help="Optionally use different devices to generate the "
                                                                   "point cloud than were used to compute the original "
                                                                   "atlas.")
    argparser.add_argument("--output", "-o", type=str, default="out.ply",
                           help="Output a dense upsampled point-cloud. The number of points per patch is 8^2 by "
                                "default and can be set by specifying --upsamples-per-patch. Default: 'out.ply'.")
    argparser.add_argument("--upsamples-per-patch", "-nup", type=int, default=8,
                           help="*Square root* of the number of upsamples per patch to generate in the output. i.e. if "
                                "you pass in --upsamples-per-patch 8, there will be 64 upsamples per patch. "
                                "Default: 8.")
    argparser.add_argument("--normal-neighborhood-size", "-ns", type=int, default=64,
                           help="")
    argparser.add_argument("--scale", type=float, default=-1.0,
                           help="Only use scale fraction of the domain [0, 1]^2 to generate points. "
                                "E.g. if scale is 0.9, then the domain for each patch is [0.05, 0.95]^2. By default, "
                                "this parameter is 1/c (where c is the padding parameter in reconstruct_surface.py)")
    argparser.add_argument("--pre-consistency", action="store_true",
                           help="Plot the reconstruction using the model generated before the consistency refinement.")
    args = argparser.parse_args()

    print("Loading state...")
    state = torch.load(args.state_file)

    devices = state["devices"]
    if args.devices:
        devices = args.devices

    print("Creating models...")
    model = nn.ModuleList([MLP(2, 3).to(devices[i % len(devices)]) for i in range(len(state["patch_idx"]))])

    if args.pre_consistency:
        model.load_state_dict(state["pre_cycle_consistency_model"])
    else:
        model.load_state_dict(state["final_model"])

    if args.scale < 0.0:
        scale = 1.0 / state["padding"]
    else:
        scale = args.scale

    print("Generating upsamples...")
    v, n = upsample_surface(state["patch_uvs"], state["patch_txs"], model, devices,
                            scale=scale, num_samples=args.upsamples_per_patch,
                            normal_samples=args.normal_neighborhood_size, compute_normals=False)

    print("Saving upsampled cloud...")
    pcu.write_ply(args.output, v, np.zeros([], dtype=np.int32), n, np.zeros([], dtype=v.dtype))

                  
if __name__ == "__main__":
    main()
