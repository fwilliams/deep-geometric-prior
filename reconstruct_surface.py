import argparse

import torch
import torch.nn as nn
import numpy as np
import point_cloud_utils as pcu

import utils
from fml.nn import SinkhornLoss
from scipy.spatial import KDTree


class MLP(nn.Module):
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

    kdtree = KDTree(x)
    ball_radius = c * r
    angle_thresh = np.cos(np.rad2deg(angle_thresh))

    patch_indexes = []
    patch_uvs = []

    for i in range(ctr_v.shape[0]):
        patch_i = np.array(kdtree.query_ball_point(ctr_v[i], ball_radius))
        good_normals = np.squeeze(n[patch_i] @ ctr_n[i].reshape([3, 1]) > angle_thresh)
        patch_i = patch_i[good_normals]

        covered_indices = patch_i[np.linalg.norm(x[patch_i] - ctr_v[i], axis=1) < r]
        covered[covered_indices] = True

        uv_i = pcu.lloyd_2d(len(patch_i)).astype(np.float32)
        patch_indexes.append(torch.from_numpy(patch_i))
        patch_uvs.append(torch.from_numpy(uv_i).to(device))

    if np.sum(covered) == x.shape[0]:
        return patch_indexes, patch_uvs
    else:
        # TODO: Handle uncovered vertices
        print("There are %d uncovered vertices" % (x.shape[0] - np.sum(covered)))
        raise NotImplementedError("compute_patches does not support uncovered vertices yet!")


def normalize_point_cloud(v, n):
    v = v.copy()
    n = n.copy()
    translate = -np.mean(v, axis=0)
    v += translate

    scale = 1.0 / np.max(np.linalg.norm(v, axis=1))
    v *= scale

    rotate, _, _ = np.linalg.svd(v.T, full_matrices=False)
    v = v @ rotate
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n = n @ rotate

    return v, n, (rotate, translate, scale)


def evaluate(uv, patch_models, scale=1.0):
    from mayavi import mlab

    with torch.no_grad():
        for i in range(len(patch_models)):
            n = 128
            uv_mesh = utils.meshgrid_from_lloyd_ts(uv[i].cpu().numpy(), n, scale=scale).astype(np.float32)
            uv_mesh = torch.from_numpy(uv_mesh).to(uv[0])
            mesh_v = patch_models[i](uv_mesh).squeeze().cpu().numpy()
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


def fit_two_phase():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("radius", type=float, help="Patch radius (The parameter, r, in the paper)")
    argparser.add_argument("padding", type=float, help="Padding factor for patches (The parameter, c, in the paper)")
    argparser.add_argument("--angle-threshold", "-a", type=float, default=95.0,
                           help="Threshold (in degrees) used to discard points in "
                                "a patch whose normal is facing the wrong way.")
    argparser.add_argument("--epochs", "-n", type=int, default=1000, help="Number of fitting iterations")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
    argparser.add_argument("--average-cycle-consistency", action="store_true",
                           help="If set, then fit all overlapping patches at the average of each point. If unset, the "
                                "fitted surface will interpolate the points.")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                           help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--sinkhorn-epsilon", "-sl", type=float, default=1e-3,
                           help="The reciprocal (1/lambda) of the sinkhorn regularization parameter.")
    argparser.add_argument("--device", "-d", type=str, default="cuda",
                           help="The device to use when fitting (either 'cpu' or 'cuda')")
    argparser.add_argument("--output", "-o", type=str, default="out.pt",
                           help="Destination to save the output reconstruction. Note, the file produced by this script "
                                "is not a mesh or a point cloud. To construct a dense point cloud, see upsample.py.")
    args = argparser.parse_args()

    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(args.mesh_filename, compute_normals=False)
    x, n, tx = normalize_point_cloud(x, n)

    # Compute a set of neighborhood (patches) and a uv samples for each neighborhood. Store the result in a list
    # of pairs (uv_j, xi_j) where uv_j are 2D uv coordinates for the j^th patch, and xi_j are the indices into x of
    # the j^th patch. We will try to reconstruct a function phi, such that phi(uv_j) = x[xi_j].
    bbox_diag = np.linalg.norm(np.max(x, axis=0) - np.min(x, axis=0))
    patch_idx, patch_uvs = compute_patches(x, n, args.radius*bbox_diag, args.padding, args.angle_threshold, args.device)
    num_patches = len(patch_uvs)
    plot_patches(x, patch_idx)

    # Initialize one model per patch and convert the input data to a pytorch tensor
    phi = nn.ModuleList([MLP(2, 3).to(args.device) for _ in range(num_patches)])
    x = torch.from_numpy(x.astype(np.float32)).to(args.device)

    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters)
    mse_loss = nn.MSELoss()

    # Fit a function, phi_i, for each patch so that phi_i(patches[i]) = x[patches[i]]. i.e. so that the function
    # phi_i "agrees" with the point cloud on each patch. The use of the Sinkhorn loss makes the fitted patches robust
    # to noisy point clouds.
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        patch_losses = []
        sum_loss = 0.0
        for i in range(num_patches):
            idx_i = patch_idx[i]
            uv_i = patch_uvs[i]
            x_i = x[idx_i]
            y_i = phi[i](uv_i)

            loss_i = sinkhorn_loss(x_i.unsqueeze(0), y_i.unsqueeze(0))
            patch_losses.append(loss_i)
            sum_loss += loss_i.item()

        for l in patch_losses:
            l.backward()

        print("%d: [Total = %0.5f] [Mean = %0.5f]" % (epoch, sum_loss, sum_loss / num_patches))
        optimizer.step()

    # Compute explicit correspondences between the uv samples for each patch and the points in the point cloud. We
    # will use these correspondences to do a second fitting stage to make the patches agree on common points.
    # The correspondences are stored in a list, pi where pi[i] is a vector of integers used to permute the points in a
    # patch.
    #
    # We also compute for each input point, the average of all the patch predictions which correspond to that input
    # point. We store these predictions in x_avg which has the same shape as x.
    #
    # Note: The final fitting makes each patch agree with this average instead of the original points. Since the
    # individual patches do not interpolate the input (a desired property to be robust to noise), we don't want the
    # second fitting stage to produce an interpolatory result.
    with torch.no_grad():
        if args.average_cycle_consistency:
            x_avg = torch.zeros_like(x)
            counts = torch.zeros(x.shape[0]).to(x)
        else:
            x_avg = x
        sinkhorn_loss.return_transport_matrix = True  # We need the transport matrix for computing correspondences
        pi = []
        for i in range(num_patches):
            idx_i = patch_idx[i]
            uv_i = patch_uvs[i]
            x_i = x[idx_i]
            y_i = phi[i](uv_i)
            _, p_i = sinkhorn_loss(y_i.unsqueeze(0), x_i.unsqueeze(0))
            pi_i = p_i.squeeze().max(0)[1]
            pi.append(pi_i)

            if args.average_cycle_consistency:
                counts_i = counts[idx_i].unsqueeze(1)
                x_avg[idx_i] = (y_i[pi_i] + counts_i * x_avg[idx_i]) / (counts_i + 1)
                counts[idx_i] += 1

    evaluate(patch_uvs, phi, scale=1.0 / args.radius)

    # Do a second, global, stage of fitting where we ask all patches to agree with each other on overlapping points.
    # This is possible since now have correspondences
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        patch_losses = []
        sum_loss = 0.0
        for i in range(num_patches):
            idx_i = patch_idx[i]
            uv_i = patch_uvs[i]
            pi_i = pi[i]
            x_i = x_avg[idx_i]
            y_i = phi[i](uv_i)

            loss_i = mse_loss(x_i.unsqueeze(0), y_i[pi_i].unsqueeze(0))
            patch_losses.append(loss_i)
            sum_loss += loss_i.item()

        for l in patch_losses:
            l.backward()

        print("%d: [Total = %0.5f] [Mean = %0.5f]" % (epoch, sum_loss, sum_loss / num_patches))
        optimizer.step()

    evaluate(patch_uvs, phi, scale=1.0 / args.radius)


if __name__ == "__main__":
    fit_two_phase()
