import argparse
import copy
import time

import numpy as np
import ot
import point_cloud_utils as pcu
import torch
import torch.nn as nn
from fml.nn import SinkhornLoss, pairwise_distances
from scipy.spatial import cKDTree

import utils


class MLP(nn.Module):
    """
    A simple fully connected network mapping vectors in dimension in_dim to vectors in dimension out_dim
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def compute_patches(x, n, r, c, angle_thresh=95.0,  min_pts_per_patch=10, devices=('cpu',)):
    """
    Given an input point cloud, X, compute a set of patches (subsets of X) and parametric samples for those patches.
    Each patch is a cluster of points which lie in a ball of radius c * r and share a similar normal.
    The spacing between patches is roughly the radius, r. This function also returns a set of 2D parametric samples
    for each patch. These samples are used to fit a function from the samples to R^3 which agrees with the patch.

    :param x: A 3D point cloud with |x| points specified as an array of shape (|x|, 3) (each row is a point)
    :param n: Unit normals for the point cloud, x, of shape (|x|, 3) (each row is a unit normal)
    :param r: The approximate separation between patches
    :param c: Each patch will fit inside a ball of radius c * r
    :param angle_thresh: If the normal of a point in a patch differs by greater than angle_thresh degrees from the
                        normal of the point at the center of the patch, it is discarded.
    :param min_pts_per_patch: The minimum number of points allowed in a patch
    :param devices: A list of devices on which to store each patch. Patch i is stored on devices[i % len(devices)].
    :return: Two lists, idx and uv, of torch tensors, where uv[i] are the parametric samples (shape = (np, 2)) for
             the i^th patch, and idx[i] are the indexes into x of the points for the i^th patch. i.e. x[idx[i]] are the
             3D points of the i^th patch.
    """

    covered = np.zeros(x.shape[0], dtype=np.bool)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ctr_v, ctr_n = pcu.prune_point_cloud_poisson_disk(x, n, r, best_choice_sampling=True)

    if len(ctr_v.shape) == 1:
        ctr_v = ctr_v.reshape([1, *ctr_v.shape])
        ctr_n = ctr_n.reshape([1, *ctr_n.shape])
    kdtree = cKDTree(x)
    ball_radius = c * r
    angle_thresh = np.cos(np.deg2rad(angle_thresh))

    patch_indexes = []
    patch_uvs = []
    patch_xs = []
    patch_transformations = []

    def make_patch(v_ctr, n_ctr):
        idx_i = np.array(kdtree.query_ball_point(v_ctr, ball_radius, p=np.inf))
        good_normals = np.squeeze(n[idx_i] @ n_ctr.reshape([3, 1]) > angle_thresh)
        idx_i = idx_i[good_normals]

        if len(idx_i) < min_pts_per_patch:
            print("Rejecting small patch with %d points" % len(idx_i))
            return

        covered_indices = idx_i[np.linalg.norm(x[idx_i] - v_ctr, axis=1) < r]
        covered[covered_indices] = True

        uv_i = pcu.lloyd_2d(len(idx_i)).astype(np.float32)
        x_i = x[idx_i].astype(np.float32)
        translate_i = -np.mean(x_i, axis=0)

        device = devices[len(patch_xs) % len(devices)]
                
        scale_i = np.array([1.0 / np.max(np.linalg.norm(x_i + translate_i, axis=1))], dtype=np.float32)
        rotate_i, _, _ = np.linalg.svd((x_i + translate_i).T, full_matrices=False)
        transform_i = (torch.from_numpy(translate_i).to(device),
                       torch.from_numpy(scale_i).to(device),
                       torch.from_numpy(rotate_i).to(device))

        x_i = torch.from_numpy((scale_i * (x_i.astype(np.float32) + translate_i)) @ rotate_i).to(device)

        patch_transformations.append(transform_i)
        patch_indexes.append(torch.from_numpy(idx_i))
        patch_uvs.append(torch.tensor(uv_i, device=device, requires_grad=True))
        patch_xs.append(x_i)
        print("Computed patch with %d points" % x_i.shape[0])
        
    for i in range(ctr_v.shape[0]):
        make_patch(ctr_v[i], ctr_n[i])

    for i in range(x.shape[0]):
        if np.sum(covered) == x.shape[0]:
            break
        if not covered[i]:
            make_patch(x[i], n[i])

    # assert np.sum(covered) == x.shape[0], "There should always be one at least one patch per input vertex"

    print("Found %d neighborhoods" % len(patch_indexes))
    return patch_indexes, patch_uvs, patch_xs, patch_transformations


def upsample_surface(patch_uvs, patch_tx, patch_models, devices, scale=1.0, num_samples=8, normal_samples=64,
                     compute_normals=True):
    vertices = []
    normals = []
    with torch.no_grad():
        for i in range(len(patch_models)):
            if (i + 1) % 10 == 0:
                print("Upsamling %d/%d" % (i+1, len(patch_models)))

            device = devices[i % len(devices)]

            n = num_samples
            translate_i, scale_i, rotate_i = (patch_tx[i][j].to(device) for j in range(len(patch_tx[i])))
            uv_i = utils.meshgrid_from_lloyd_ts(patch_uvs[i].cpu().numpy(), n, scale=scale).astype(np.float32)
            uv_i = torch.from_numpy(uv_i).to(patch_uvs[i])
            y_i = patch_models[i](uv_i)

            mesh_v = ((y_i.squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i).cpu().numpy()

            if compute_normals:
                mesh_f = utils.meshgrid_face_indices(n)
                mesh_n = pcu.per_vertex_normals(mesh_v, mesh_f)
                normals.append(mesh_n)

            vertices.append(mesh_v)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    if compute_normals:
        normals = np.concatenate(normals, axis=0).astype(np.float32)
    else:
        print("Fixing normals...")
        normals = pcu.estimate_normals(vertices, k=normal_samples)

    return vertices, normals


def plot_reconstruction(x, patch_uvs, patch_tx, patch_models, scale=1.0):
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

        mlab.points3d(x[:, 0], x[:, 1], x[:, 2], scale_factor=0.001)
        mlab.show()


def plot_patches(x, patch_idx):
    from mayavi import mlab

    mlab.figure(bgcolor=(1, 1, 1))
    for idx_i in patch_idx:
        color = tuple(np.random.rand(3))
        sf = 0.1 + np.random.randn()*0.05
        x_i = x[idx_i]
        mlab.points3d(x_i[:, 0], x_i[:, 1], x_i[:, 2], color=color, scale_factor=sf)
    mlab.show()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("radius", type=float, help="Patch radius (The parameter, r, in the paper)")
    argparser.add_argument("padding", type=float, help="Padding factor for patches (The parameter, c, in the paper)")
    argparser.add_argument("min_pts_per_patch", type=int,
                           help="Minimum number of allowed points inside a patch used to not fit to "
                                "patches with too little data")
    argparser.add_argument("--output", "-o", type=str, default="out.ply",
                           help="Output a dense upsampled point-cloud. The number of points per patch is 8^2 by "
                                "default and can be set by specifying --upsamples-per-patch.")
    argparser.add_argument("--out-meta", "-om", type=str, default="out.pt",
                           help="Destination to save the output reconstruction metadata. "
                                "Note, the file specified here is *not* a mesh or a point cloud. "
                                "To construct a dense point cloud, see --out-mesh and export_point_cloud.py.")
    argparser.add_argument("--upsamples-per-patch", "-nup", type=int, default=8,
                           help="*Square root* of the number of upsamples per patch to generate in the output. i.e. if "
                                "you pass in --upsamples-per-patch 8, there will be 64 upsamples per patch.")
    argparser.add_argument("--angle-threshold", "-a", type=float, default=95.0,
                           help="Threshold (in degrees) used to discard points in "
                                "a patch whose normal is facing the wrong way.")
    argparser.add_argument("--local-epochs", "-nl", type=int, default=128,
                           help="Number of fitting iterations done for each chart to its points")
    argparser.add_argument("--global-epochs", "-ng", type=int, default=128,
                           help="Number of fitting iterations done to make each chart agree "
                                "with its neighboring charts")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3,
                           help="Step size for gradient descent.")
    argparser.add_argument("--devices", "-d", type=str, default=["cuda"], nargs="+",
                           help="A list of devices on which to partition the models for each patch. For large inputs, "
                                "reconstruction can be memory and compute intensive. Passing in multiple devices will "
                                "split the load across these. E.g. --devices cuda:0 cuda:1 cuda:2")
    argparser.add_argument("--plot", action="store_true",
                           help="Plot the following intermediate states:. (1) patch neighborhoods, "
                                "(2) Intermediate reconstruction before global consistency step, "
                                "(3) Reconstruction after global consistency step. "
                                "This flag is useful for debugging but does not scale well to large inputs.")
    argparser.add_argument("--interpolate", action="store_true",
                           help="If set, then force all patches to agree with the input at overlapping points "
                                "(i.e. the reconstruction will try to interpolate the input point cloud). "
                                "Otherwise, we fit all patches to the average of overlapping patches at each point.")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                           help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--sinkhorn-epsilon", "-sl", type=float, default=1e-3,
                           help="The reciprocal (1/lambda) of the Sinkhorn regularization parameter.")
    argparser.add_argument("--seed", "-s", type=int, default=-1,
                           help="Random seed to use when initializing network weights. "
                                "If the seed not positive, a seed is selected at random.")
    argparser.add_argument("--exact-emd", "-e", action="store_true",
                           help="Use exact optimal transport distance instead of sinkhorn. "
                                "This will be slow and should not make a difference in the output")
    argparser.add_argument("--use-best", action="store_true",
                           help="Use the model with the lowest loss as the final result.")
    argparser.add_argument("--normal-neighborhood-size", "-ns", type=int, default=64,
                           help="Neighborhood size used to estimate the normals in the final dense point cloud. "
                                "Default: 64")

    args = argparser.parse_args()

    # We'll populate this dictionary and save it as output
    output_dict = {
        "pre_cycle_consistency_model": None,
        "final_model": None,
        "patch_uvs": None,
        "patch_idx": None,
        "patch_txs": None,
        "radius": args.radius,
        "padding": args.padding,
        "min_pts_per_patch": args.min_pts_per_patch,
        "angle_threshold": args.angle_threshold,
        "interpolate": args.interpolate,
        "global_epochs": args.global_epochs,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "devices": args.devices,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "max_sinkhorn_iters": args.max_sinkhorn_iters,
        "seed": utils.seed_everything(args.seed),
    }

    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(args.mesh_filename, compute_normals=True)

    # Compute a set of neighborhood (patches) and a uv samples for each neighborhood. Store the result in a list
    # of pairs (uv_j, xi_j) where uv_j are 2D uv coordinates for the j^th patch, and xi_j are the indices into x of
    # the j^th patch. We will try to reconstruct a function phi, such that phi(uv_j) = x[xi_j].
    print("Computing neighborhoods...")
    bbox_diag = np.linalg.norm(np.max(x, axis=0) - np.min(x, axis=0))
    patch_idx, patch_uvs, patch_xs, patch_tx = compute_patches(x, n, args.radius*bbox_diag, args.padding,
                                                               angle_thresh=args.angle_threshold,
                                                               min_pts_per_patch=args.min_pts_per_patch,
                                                               devices=args.devices)
    num_patches = len(patch_uvs)
    output_dict["patch_uvs"] = patch_uvs
    output_dict["patch_idx"] = patch_idx
    output_dict["patch_txs"] = patch_tx

    if args.plot:
        plot_patches(x, patch_idx)

    # Initialize one model per patch and convert the input data to a pytorch tensor
    print("Creating models...")
    phi = nn.ModuleList([MLP(2, 3).to(args.devices[i % len(args.devices)]) for i in range(num_patches)])
    # x = torch.from_numpy(x.astype(np.float32)).to(args.device)

    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    uv_optimizer = torch.optim.Adam(patch_uvs, lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters, return_transport_matrix=True)
    mse_loss = nn.MSELoss()

    # Fit a function, phi_i, for each patch so that phi_i(patch_uvs[i]) = x[patch_idx[i]]. i.e. so that the function
    # phi_i "agrees" with the point cloud on each patch.
    #
    # We also store the correspondences between the uvs and points which we use later for the consistency step. The
    # correspondences are stored in a list, pi where pi[i] is a vector of integers used to permute the points in
    # a patch.
    pi = [None for _ in range(num_patches)]

    # Cache model with the lowest loss if --use-best is passed
    best_models = [None for _ in range(num_patches)]
    best_losses = [np.inf for _ in range(num_patches)]

    for epoch in range(args.local_epochs):
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        sum_loss = torch.tensor([0.0]).to(args.devices[0])
        epoch_start_time = time.time()
        for i in range(num_patches):
            uv_i = patch_uvs[i]
            x_i = patch_xs[i]
            y_i = phi[i](uv_i)

            with torch.no_grad():
                if args.exact_emd:
                    M_i = pairwise_distances(x_i.unsqueeze(0), y_i.unsqueeze(0)).squeeze().cpu().squeeze().numpy()
                    p_i = ot.emd(np.ones(x_i.shape[0]), np.ones(y_i.shape[0]), M_i)
                    p_i = torch.from_numpy(p_i.astype(np.float32)).to(args.devices[0])
                else:
                    _, p_i = sinkhorn_loss(x_i.unsqueeze(0), y_i.unsqueeze(0))
                pi_i = p_i.squeeze().max(0)[1]
                pi[i] = pi_i

            loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))

            if args.use_best and loss_i.item() < best_losses[i]:
                best_losses[i] = loss_i
                best_models[i] = copy.deepcopy(phi[i].state_dict())

            sum_loss += loss_i.to(args.devices[0])

        sum_loss.backward()
        epoch_end_time = time.time()

        print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
              (epoch, args.local_epochs, sum_loss.item(),
               sum_loss.item() / num_patches, epoch_end_time-epoch_start_time))
        optimizer.step()
        uv_optimizer.step()

    if args.use_best:
        for i, phi_i in enumerate(phi):
            phi_i.load_state_dict(best_models[i])

    output_dict["pre_cycle_consistency_model"] = copy.deepcopy(phi.state_dict())

    if args.plot:
        plot_reconstruction(x, patch_uvs, patch_tx, phi, scale=1.0/args.padding)

    # Do a second, global, stage of fitting where we ask all patches to agree with each other on overlapping points.
    # If the user passed --interpolate, we ask that the patches agree on the original input points, otherwise we ask
    # that they agree on the average of predictions from patches overlapping a given point.
    for epoch in range(args.global_epochs):
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        sum_loss = torch.tensor([0.0]).to(args.devices[0])
        epoch_start_time = time.time()
        for i in range(num_patches):
            uv_i = patch_uvs[i]
            x_i = patch_xs[i]
            y_i = phi[i](uv_i)
            pi_i = pi[i]
            loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))

            if loss_i.item() < best_losses[i]:
                best_losses[i] = loss_i
                best_models[i] = copy.deepcopy(phi[i].state_dict())

            sum_loss += loss_i.to(args.devices[0])

        sum_loss.backward()
        epoch_end_time = time.time()

        print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
              (epoch, args.global_epochs, sum_loss.item(),
               sum_loss.item() / num_patches, epoch_end_time-epoch_start_time))
        optimizer.step()
        uv_optimizer.step()

    for i, phi_i in enumerate(phi):
        phi_i.load_state_dict(best_models[i])

    output_dict["final_model"] = copy.deepcopy(phi.state_dict())

    print("Generating dense point cloud...")
    v, n = upsample_surface(patch_uvs, patch_tx, phi, args.devices,
                            scale=1.0/args.padding,
                            normal_samples=args.normal_neighborhood_size,
                            num_samples=args.upsamples_per_patch,
                            compute_normals=False)
    print("Saving dense point cloud...")
    pcu.write_ply(args.output, v, np.zeros([], dtype=np.int32), n, np.zeros([], dtype=v.dtype))

    print("Saving metadata...")
    torch.save(output_dict, args.out_meta)

    if args.plot:
        plot_reconstruction(x, patch_uvs, patch_tx, phi, scale=1.0/args.padding)


if __name__ == "__main__":
    main()
