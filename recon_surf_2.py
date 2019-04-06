import argparse
import copy
import time

import torch
import torch.nn as nn
import numpy as np
import point_cloud_utils as pcu

import utils
from fml.nn import SinkhornLoss
from scipy.spatial import KDTree


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
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class MLPUltraShallow(nn.Module):
    """
    A single hidden-layer fully-connected network mapping vectors in dimension in_dim to vectors in dimension out_dim
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_patches(x, n, r, c, angle_thresh=95.0, device='cpu'):
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
    :param device: The device on which the patches are stored
    :return: Two lists, idx and uv, of torch tensors, where uv[i] are the parametric samples (shape = (np, 2)) for
             the i^th patch, and idx[i] are the indexes into x of the points for the i^th patch. i.e. x[idx[i]] are the
             3D points of the i^th patch.
    """

    covered = np.zeros(x.shape[0], dtype=np.bool)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ctr_v, ctr_n = pcu.sample_point_cloud_poisson_disk(x, n, r, best_choice_sampling=True)

    if len(ctr_v.shape) == 1:
        ctr_v = ctr_v.reshape([1, *ctr_v.shape])
        ctr_n = ctr_n.reshape([1, *ctr_n.shape])
    kdtree = KDTree(x)
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

        if len(idx_i) == 0:
            return

        covered_indices = idx_i[np.linalg.norm(x[idx_i] - v_ctr, axis=1) < r]
        covered[covered_indices] = True

        uv_i = pcu.lloyd_2d(len(idx_i)).astype(np.float32)
        x_i = x[idx_i].astype(np.float32)
        translate_i = -np.mean(x_i, axis=0)
        scale_i = np.array([1.0 / np.max(np.linalg.norm(x_i + translate_i, axis=1))], dtype=np.float32)
        rotate_i, _, _ = np.linalg.svd((x_i + translate_i).T, full_matrices=False)
        transform_i = (torch.from_numpy(translate_i).to(device),
                       torch.from_numpy(scale_i).to(device),
                       torch.from_numpy(rotate_i).to(device))

        x_i = torch.from_numpy((scale_i * (x_i.astype(np.float32) + translate_i)) @ rotate_i).to(device)

        patch_transformations.append(transform_i)
        patch_indexes.append(torch.from_numpy(idx_i))
        patch_uvs.append(torch.from_numpy(uv_i).to(device))
        patch_xs.append(x_i)

    for i in range(ctr_v.shape[0]):
        make_patch(ctr_v[i], ctr_n[i])

    for i in range(x.shape[0]):
        if np.sum(covered) == x.shape[0]:
            break
        if not covered[i]:
            make_patch(x[i], n[i])

    assert np.sum(covered) == x.shape[0], "There should always be one at least one patch per input vertex"

    print("Found %d neighborhoods" % len(patch_indexes))
    return patch_indexes, patch_uvs, patch_xs, patch_transformations


def evaluate(patch_uvs, patch_tx, patch_models, scale=1.0):
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


def plot_patches(x, patch_idx):
    from mayavi import mlab

    for idx_i in patch_idx:
        color = tuple(np.random.rand(3))
        sf = 0.05 + np.random.randn()*0.005
        x_i = x[idx_i]
        mlab.points3d(x_i[:, 0], x_i[:, 1], x_i[:, 2], color=color, scale_factor=sf)
    mlab.show()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("radius", type=float, help="Patch radius (The parameter, r, in the paper)")
    argparser.add_argument("padding", type=float, help="Padding factor for patches (The parameter, c, in the paper)")
    argparser.add_argument("--angle-threshold", "-a", type=float, default=95.0,
                           help="Threshold (in degrees) used to discard points in "
                                "a patch whose normal is facing the wrong way.")
    argparser.add_argument("--local-epochs", "-nl", type=int, default=512, help="Number of local fitting iterations")
    argparser.add_argument("--global-epochs", "-ng", type=int, default=1024, help="Number of global fitting iterations")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
    argparser.add_argument("--device", "-d", type=str, default="cuda",
                           help="The device to use when fitting (either 'cpu' or 'cuda')")
    argparser.add_argument("--interpolate", action="store_true",
                           help="If set, then force all patches to agree with the input at overlapping points. "
                                "Otherwise, we fit all patches to the average of overlapping patches at each point.")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                           help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--sinkhorn-epsilon", "-sl", type=float, default=1e-3,
                           help="The reciprocal (1/lambda) of the sinkhorn regularization parameter.")
    argparser.add_argument("--output", "-o", type=str, default="out.pt",
                           help="Destination to save the output reconstruction. Note, the file produced by this script "
                                "is not a mesh or a point cloud. To construct a dense point cloud, see upsample.py.")
    argparser.add_argument("--seed", "-s", type=int, default=-1,
                           help="Random seed to use when initializing network weights. "
                                "If the seed not positive, a seed is selected at random.")
    args = argparser.parse_args()

    # We'll populate this dictionary and save it as output
    output_dict = {
        "pre_cycle_consistency_model": None,
        "final_model": None,
        "patch_uvs": None,
        "patch_idx": None,
        "patch_txs": None,
        "interpolate": args.interpolate,
        "global_epochs": args.global_epochs,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "max_sinkhorn_iters": args.max_sinkhorn_iters,
        "seed": utils.seed_everything(args.seed),
    }

    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(args.mesh_filename, compute_normals=True)
    # x, n, tx = normalize_point_cloud(x, n)

    # Compute a set of neighborhood (patches) and a uv samples for each neighborhood. Store the result in a list
    # of pairs (uv_j, xi_j) where uv_j are 2D uv coordinates for the j^th patch, and xi_j are the indices into x of
    # the j^th patch. We will try to reconstruct a function phi, such that phi(uv_j) = x[xi_j].
    bbox_diag = np.linalg.norm(np.max(x, axis=0) - np.min(x, axis=0))
    patch_idx, patch_uvs, patch_xs, patch_tx = compute_patches(x, n, args.radius*bbox_diag, args.padding,
                                                               args.angle_threshold, args.device)
    num_patches = len(patch_uvs)
    output_dict["patch_uvs"] = patch_uvs
    output_dict["patch_idx"] = patch_idx
    output_dict["patch_txs"] = patch_tx
    # plot_patches(x, patch_idx)

    # Initialize one model per patch and convert the input data to a pytorch tensor
    phi = nn.ModuleList([MLP(2, 3).to(args.device) for _ in range(num_patches)])
    # x = torch.from_numpy(x.astype(np.float32)).to(args.device)

    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters, return_transport_matrix=True)
    mse_loss = nn.MSELoss()

    # Fit a function, phi_i, for each patch so that phi_i(patch_uvs[i]) = x[patch_idx[i]]. i.e. so that the function
    # phi_i "agrees" with the point cloud on each patch. The use of the Sinkhorn loss makes the fitted patches robust
    # to noisy point clouds.
    #
    # We also store the correspondences between the uvs and points which we use later for the cycle consistency. The
    # correspondences are stored in a list, pi where pi[i] is a vector of integers used to permute the points in
    # a patch.
    pi = [None for _ in range(num_patches)]
    for epoch in range(args.local_epochs):
        optimizer.zero_grad()

        # patch_losses = []
        sum_loss = torch.tensor([0.0]).to(args.device)
        epoch_start_time = time.time()
        for i in range(num_patches):
            uv_i = patch_uvs[i]
            x_i = patch_xs[i]
            y_i = phi[i](uv_i)

            with torch.no_grad():
                _, p_i = sinkhorn_loss(y_i.unsqueeze(0), x_i.unsqueeze(0))
                pi_i = p_i.squeeze().max(0)[1]
                pi[i] = pi_i

            sum_loss += mse_loss(x_i.unsqueeze(0), y_i[pi_i].unsqueeze(0))

            # loss_i, p_i = sinkhorn_loss(y_i.unsqueeze(0), x_i.unsqueeze(0))
            # pi_i = p_i.detach().squeeze().max(0)[1]
            # pi[i] = pi_i

        sum_loss.backward()
        epoch_end_time = time.time()

        print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
              (epoch, args.local_epochs, sum_loss.item(),
               sum_loss.item() / num_patches, epoch_end_time-epoch_start_time))
        optimizer.step()

    output_dict["pre_cycle_consistency_model"] = copy.deepcopy(phi.state_dict())

    evaluate(patch_uvs, patch_tx, phi, scale=0.9)

    # If the user does not specify --interpolate, we compute for each input point, the average of all the patch
    # predictions which correspond to it. We store these predictions in x_avg which has the same shape as x.
    # if not args.interpolate:
    #     with torch.no_grad():
    #         x_avg = torch.zeros_like(x)
    #         counts = torch.zeros(x.shape[0]).to(x)
    #         for i in range(num_patches):
    #             idx_i = patch_idx[i]
    #             uv_i = patch_uvs[i]
    #             y_i = phi[i](uv_i)
    #             pi_i = pi[i]
    #             counts_i = counts[idx_i].unsqueeze(1)
    #             x_avg[idx_i] = (y_i[pi_i] + counts_i * x_avg[idx_i]) / (counts_i + 1)
    #             counts[idx_i] += 1
    # else:
    #     x_avg = x

    # Do a second, global, stage of fitting where we ask all patches to agree with each other on overlapping points.
    # If the user passed --interpolate, we ask that the patches agree on the original input points, otherwise we ask
    # that they agree on the average of predictions from patches overlapping a given point.
    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    for epoch in range(args.global_epochs):
        optimizer.zero_grad()

        # patch_losses = []
        sum_loss = torch.tensor([0.0]).to(args.device)
        epoch_start_time = time.time()
        for i in range(num_patches):
            uv_i = patch_uvs[i]
            x_i = patch_xs[i]
            y_i = phi[i](uv_i)
            pi_i = pi[i]
            sum_loss += mse_loss(x_i.unsqueeze(0), y_i[pi_i].unsqueeze(0))

        sum_loss.backward()
        epoch_end_time = time.time()

        print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
              (epoch, args.global_epochs, sum_loss.item(),
               sum_loss.item() / num_patches, epoch_end_time-epoch_start_time))
        optimizer.step()

    output_dict["final_model"] = copy.deepcopy(phi.state_dict())

    evaluate(patch_uvs, patch_tx, phi, scale=0.9)

    torch.save(output_dict, args.output)


if __name__ == "__main__":
    main()
