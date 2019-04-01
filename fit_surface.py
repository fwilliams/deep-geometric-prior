import argparse

import torch
import torch.nn as nn
import numpy as np

import utils
import models
from fml.nn import SinkhornLoss
from point_cloud_utils import lloyd_2d, sample_mesh_poisson_disk
from scipy.spatial import KDTree


def compute_patches(X, N, r, alpha, back_thresh=-0.3, device='cpu'):
    covered = np.zeros(X.shape[0], dtype=np.bool)
    ctr_v, ctr_n = sample_mesh_poisson_disk(X, np.zeros([0, 3], dtype=np.int32), N, r,
                                            use_geodesic_distance=False, best_choice_sampling=True)
    # ctr_v, ctr_n = poisson_disk_sample(X, np.zeros([0, 3], dtype=np.int32), N, r,
    #                                    use_geodesic_distance=False,
    #                                    best_choice_sampling=True)

    kdtree = KDTree(X)
    ball_radius = r*alpha

    patches = []

    for i in range(ctr_v.shape[0]):
        patch_i = np.array(kdtree.query_ball_point(ctr_v[i], ball_radius))
        good_normals = np.squeeze(N[patch_i] @ ctr_n[i].reshape([3, 1]) > back_thresh)
        patch_i = patch_i[good_normals]

        covered_indices = patch_i[np.linalg.norm(X[patch_i] - ctr_v[i], axis=1) < r]
        covered[covered_indices] = True

        uv_i = lloyd_2d(len(patch_i)).astype(np.float32)
        patches.append((torch.from_numpy(uv_i).to(device), torch.from_numpy(patch_i)))

    if np.sum(covered) == X.shape[0]:
        return patches
    else:
        raise NotImplementedError("TODO: Handle uncovered vertices")


def step_local(X, Y, P, return_transport_matrices=False, max_sinkhorn_iters=300, reduce=True):
    sinkhorn_fun = SinkhornLoss(max_iters=max_sinkhorn_iters, return_transport_matrix=return_transport_matrices)

    loss = torch.tensor([0]).to(X) if reduce else []
    Pi = []

    for i in range(len(P)):
        c_i = P[i][1]

        loss_i = sinkhorn_fun(Y[i].unsqueeze(0), X[c_i].unsqueeze(0))
        if return_transport_matrices:
            loss_i, pi_i = loss_i
            Pi.append(pi_i.squeeze().max(0)[1].cpu())

        if reduce:
            loss += loss_i
        else:
            loss.append(loss_i)

    return (loss, Pi) if return_transport_matrices else loss


def step_global(X_guess, Y, Pi, P, reduce=True):
    loss = torch.tensor([0]).to(X_guess) if reduce else []
    for i in range(len(Y)):
        c_i = P[i][1]
        pi_i = Pi[i]
        loss_i = (Y[i][pi_i] - X_guess[c_i]).pow(2).sum(1).sum(0)
        if reduce:
            loss += loss_i
        else:
            loss.append(loss_i)
    return loss


def correspondences(X, Y, P, max_sinkhorn_iters=300):
    with torch.no_grad():
        sinkhorn_fun = SinkhornLoss(max_iters=max_sinkhorn_iters, return_transport_matrix=True)

        Pi = []

        for i in range(len(P)):
            c_i = P[i][1]

            _, pi_i = sinkhorn_fun(Y[i].unsqueeze(0), X[c_i].unsqueeze(0))
            Pi.append(pi_i.squeeze().max(0)[1].cpu())

    return Pi


def patch_averages(X, Y, Pi, P):
    with torch.no_grad():
        X_guess = torch.zeros(X.shape).to(X)
        counts = torch.zeros(X.shape[0]).to(X)

        for i in range(len(P)):
            c_i = P[i][1]
            n_i = counts[c_i].unsqueeze(1)
            pi_i = Pi[i]
            X_guess[c_i] = (Y[i][pi_i] + n_i * X_guess[c_i]) / (n_i + 1)

    return X_guess


def normalize_input(v, n):
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
            mesh_v = patch_models[i](None, uv_mesh).squeeze().cpu().numpy()
            mesh_f = utils.meshgrid_face_indices(n)
            mlab.triangular_mesh(mesh_v[:, 0], mesh_v[:, 1], mesh_v[:, 2], mesh_f, color=(0.2, 0.2, 0.8))
        mlab.show()


def plot_patches(X, P):
    from mayavi import mlab

    for _, c_i in P:
        color = tuple(np.random.rand(3))
        sf = 0.05 + np.random.randn()*0.005
        X_i = X[c_i]
        mlab.points3d(X_i[:, 0], X_i[:, 1], X_i[:, 2], color=color, scale_factor=sf, opacity=0.9)
    mlab.show()


def fit_two_phase():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("radius", type=float, help="Patch radius")
    argparser.add_argument("alpha", type=float, help="Padding factor for patches")
    argparser.add_argument("--epochs", "-ne", type=int, default=1000, help="Number of fitting iterations")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                           help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
    args = argparser.parse_args()

    device = 'cuda'

    X, _, N = utils.load_mesh_by_file_extension(args.mesh_filename, compute_normals=True)
    X, N, tx = normalize_input(X, N)
    bbox_diag = np.linalg.norm(np.max(X, axis=0) - np.min(X, axis=0))

    P = compute_patches(X, N, args.radius*bbox_diag, args.alpha, device=device)
    # plot_patches(X, P)
    print("Fitting %d patches" % len(P))

    Phi = nn.ModuleList([models.DerpNN(256, 256).to(device) for _ in range(len(P))])
    X = torch.from_numpy(X.astype(np.float32)).to(device)
    Y = None

    optimizer = torch.optim.Adam(Phi.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        Y = [Phi[i](None, P[i][0]).squeeze() for i in range(len(P))]

        loss = step_local(X, Y, P, max_sinkhorn_iters=32, reduce=False)
        for l in loss:
            l.backward()

        print("%d: [Total = %0.5f] [Mean = %0.5f]" % (epoch, np.sum([l.item() for l in loss]),
                                                      np.mean([l.item() for l in loss])))

        optimizer.step()

    torch.save((X, P, Y, Phi), "traindata.pt")

    # X, P, Y, Phi = torch.load("traindata.pt")
    Pi = correspondences(X, Y, P, max_sinkhorn_iters=args.max_sinkhorn_iters)

    optimizer = torch.optim.Adam(Phi.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        Y = [Phi[i](None, P[i][0]).squeeze() for i in range(len(P))]
        loss = step_global(X, Y, Pi, P, reduce=False)
        for l in loss:
            l.backward()

        print("%d: [Total = %0.5f] [Mean = %0.5f]" % (epoch, np.sum([l.item() for l in loss]),
                                                      np.mean([l.item() for l in loss])))

        optimizer.step()

    # X_est = patch_point_estimates(X, Y, Pi, P)
    # mlab.points3d(X[:, 0], X[:, 1], X[:, 2], scale_factor=0.0025, color=(0.2, 0.2, 0.8))
    # mlab.points3d(X_est[:, 0], X_est[:, 1], X_est[:, 2], scale_factor=0.02, color=(0.8, 0.2, 0.2))
    # mlab.show()

    evaluate([uv for uv, _ in P], Phi, scale=1.0/args.alpha)


def fit_one_phase():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("radius", type=float, help="Patch radius")
    argparser.add_argument("alpha", type=float, help="Padding factor for patches")
    argparser.add_argument("--epochs", "-ne", type=int, default=1000, help="Number of fitting iterations")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                           help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
    args = argparser.parse_args()

    device = 'cuda'

    X, _, N = utils.load_mesh_by_file_extension(args.mesh_filename, compute_normals=True)
    X, N, tx = normalize_input(X, N)
    bbox_diag = np.linalg.norm(np.max(X, axis=0) - np.min(X, axis=0))

    P = compute_patches(X, N, args.radius*bbox_diag, args.alpha, device=device)
    plot_patches(X, P)
    print("Fitting %d patches" % len(P))

    Phi = nn.ModuleList([models.DerpNN(256, 256).to(device) for _ in range(len(P))])
    X = torch.from_numpy(X.astype(np.float32)).to(device)
    Y = None

    optimizer = torch.optim.Adam(Phi.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        Y = [Phi[i](None, P[i][0]).squeeze() for i in range(len(P))]

        loss_local, Pi = step_local(X, Y, P, max_sinkhorn_iters=32, return_transport_matrices=True)
        X_est = patch_averages(X, Y, Pi, P)
        loss_global = step_global(X_est, Y, Pi, P)
        loss = loss_local + 0.1*loss_global

        print("%d: [Total = %0.5f] [Mean = %0.5f]" % (epoch, loss.item(), loss.item() / len(Y)))

        loss.backward()
        optimizer.step()

    torch.save((X, P, Y, Phi), "traindata.pt")

    evaluate([uv for uv, _ in P], Phi, scale=1.0/args.alpha)


if __name__ == "__main__":
    fit_two_phase()
