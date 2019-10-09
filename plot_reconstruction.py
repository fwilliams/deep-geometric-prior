import argparse

import numpy as np
import torch
import torch.nn as nn
from mayavi import mlab

import utils
from reconstruct_surface import MLP


def plot_reconstruction(patch_uvs, patch_tx, patch_models, scale=1.0):
    from mayavi import mlab

    with torch.no_grad():
        for i in range(len(patch_models)):
            n = 128
            translate_i, scale_i, rotate_i = patch_tx[i]
            uv_i = utils.meshgrid_from_lloyd_ts(patch_uvs[i].cpu().numpy(), n, scale=scale).astype(np.float32)
            uv_i = torch.from_numpy(uv_i).to(patch_uvs[0])
            y_i = patch_models[i](uv_i)

            mesh_v = ((y_i.squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i).cpu().numpy()
            mesh_f = utils.meshgrid_face_indices(n)
            mlab.triangular_mesh(mesh_v[:, 0], mesh_v[:, 1], mesh_v[:, 2], mesh_f, color=(0.2, 0.2, 0.8))

        mlab.show()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("state_file", type=str, help="Path to a reconstructed surface state file generated with "
                                                        "`reconstruct_surface.py` or `reconstruct_single_patch.py`")
    argparser.add_argument("--scale", type=float, default=-1.0)
    argparser.add_argument("--pre-consistency", action="store_true",
                           help="Plot the reconstruction using the model generated before the consistency refinement")
    args = argparser.parse_args()

    state = torch.load(args.state_file)
    model = nn.ModuleList([MLP(2, 3).to(state["device"]) for _ in range(len(state["patch_idx"]))])

    if args.pre_consistency:
        model.load_state_dict(state["pre_cycle_consistency_model"])
    else:
        model.load_state_dict(state["final_model"])

    if args.scale < 0.0:
        scale = 1.0 / state["padding"]
    else:
        scale = args.scale
    plot_reconstruction(state["patch_uvs"], state["patch_txs"], model, scale=scale)


if __name__ == "__main__":
    main()