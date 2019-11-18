import argparse

import numpy as np
import torch
import torch.nn as nn

import utils
from reconstruct_surface import MLP
import point_cloud_utils as pcu

def upsample_surface(patch_uvs, patch_tx, patch_models, scale=1.0, num_samples=128):
    vertices = []
    normals = []
    with torch.no_grad():
        for i in range(len(patch_models)):
            if (i + 1) % 10 == 0:
                print("Upsamling %d/%d" % (i+1, len(patch_models)))
            n = num_samples
            translate_i, scale_i, rotate_i = patch_tx[i]
            uv_i = utils.meshgrid_from_lloyd_ts(patch_uvs[i].cpu().numpy(), n, scale=scale).astype(np.float32)
            uv_i = torch.from_numpy(uv_i).to(patch_uvs[0])
            y_i = patch_models[i](uv_i)

            mesh_v = ((y_i.squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i).cpu().numpy()
            mesh_f = utils.meshgrid_face_indices(n)
            mesh_n = pcu.per_vertex_normals(mesh_v, mesh_f)

            vertices.append(mesh_v)
            normals.append(mesh_n)

        
    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    normals = np.concatenate(vertices, axis=0).astype(np.float32)

    return vertices, normals


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("state_file", type=str, help="Path to a reconstructed surface state file generated with "
                                                        "`reconstruct_surface.py` or `reconstruct_single_patch.py`")
    argparser.add_argument("--scale", type=float, default=-1.0)
    argparser.add_argument("--pre-consistency", action="store_true",
                           help="Plot the reconstruction using the model generated before the consistency refinement")
    argparser.add_argument("--samples-per-patch", "-n", type=int, default=128, help="Number of upsamples per patch")
    args = argparser.parse_args()

    print("Loading state...")
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

    print("Generating upsamples...")
    v, n = upsample_surface(state["patch_uvs"], state["patch_txs"], model, scale=scale,
                           num_samples=args.samples_per_patch)

    pcu.write_ply("out.ply", v, np.zeros([0, 3], dtype=np.int32), n, np.zeros([1, 2], dtype=v.dtype))

                  
if __name__ == "__main__":
    main()
