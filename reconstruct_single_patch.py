import argparse
import copy
import time

import torch
import torch.nn as nn
import numpy as np
import point_cloud_utils as pcu

import utils
from fml.nn import SinkhornLoss, pairwise_distances
import ot


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
        self.fc5 = nn.Linear(512, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def transform_pointcloud(x, device):
    translate = -np.mean(x, axis=0)
    scale = np.array([1.0 / np.max(np.linalg.norm(x + translate, axis=1))], dtype=np.float32)
    rotate, _, _ = np.linalg.svd((x + translate).T, full_matrices=False)
    transform = (torch.from_numpy(translate).to(device),
                 torch.from_numpy(scale).to(device),
                 torch.from_numpy(rotate).to(device))
    x_tx = torch.from_numpy((scale * (x.astype(np.float32) + translate)) @ rotate).to(device)

    return x_tx, transform


def plot_reconstruction(uv, x, transform, model, pad=1.0):
    from mayavi import mlab

    with torch.no_grad():
        n = 128
        translate, scale, rotate = transform
        uv_dense = utils.meshgrid_from_lloyd_ts(uv.cpu().numpy(), n, scale=pad).astype(np.float32)
        uv_dense = torch.from_numpy(uv_dense).to(uv)
        y_dense = model(uv_dense)

        # x = ((x.squeeze() @ rotate.transpose(0, 1)) / scale - translate).cpu().numpy()
        # mesh_v = ((y_dense.squeeze() @ rotate.transpose(0, 1)) / scale - translate).cpu().numpy()
        x = x.squeeze().cpu().numpy()
        mesh_v = y_dense.squeeze().cpu().numpy()
        mesh_f = utils.meshgrid_face_indices(n)

        mlab.points3d(x[:, 0], x[:, 1], x[:, 2], scale_factor=0.01)
        mlab.triangular_mesh(mesh_v[:, 0], mesh_v[:, 1], mesh_v[:, 2], mesh_f, color=(0.2, 0.2, 0.8))
        mlab.show()


def plot_correspondences(model, uv, x, pi):
    y = model(uv).detach().squeeze().cpu().numpy()

    from mayavi import mlab
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.points3d(x[:, 0], x[:, 1], x[:, 2], color=(1, 0, 0), scale_factor=0.01)
    mlab.points3d(y[:, 0], y[:, 1], y[:, 2], color=(0, 1, 0), scale_factor=0.01)
    x = x[pi].detach().squeeze().cpu().numpy()

    for i in range(x.shape[0]):
        lx = [x[i, 0], y[i, 0]]
        ly = [x[i, 1], y[i, 1]]
        lz = [x[i, 2], y[i, 2]]

        mlab.plot3d(lx, ly, lz, color=(0.1, 0.1, 0.1), tube_radius=None)
    mlab.show()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("--plot", action="store_true", help="Plot the output when done training")
    argparser.add_argument("--local-epochs", "-nl", type=int, default=512, help="Number of local fitting iterations")
    argparser.add_argument("--global-epochs", "-ng", type=int, default=1024, help="Number of global fitting iterations")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
    argparser.add_argument("--device", "-d", type=str, default="cuda",
                           help="The device to use when fitting (either 'cpu' or 'cuda')")
    argparser.add_argument("--exact-emd", "-e", action="store_true",
                           help="Use exact optimal transport distance instead of sinkhorn")
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
    argparser.add_argument("--use-best", action="store_true", help="Use the model with the lowest loss")
    argparser.add_argument("--print-every", type=int, default=16, help="Print every N epochs")
    args = argparser.parse_args()

    # We'll populate this dictionary and save it as output
    output_dict = {
        "final_model": None,
        "uv": None,
        "x": None,
        "transform": None,
        "exact_emd": args.exact_emd,
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

    # Center the point cloud about its mean and align about its principle components
    x, transform = transform_pointcloud(x, args.device)

    # Generate an initial set of UV samples in the plane
    uv = torch.tensor(pcu.lloyd_2d(x.shape[0]).astype(np.float32), requires_grad=True, device=args.device)

    # Initialize the model for the surface
    # phi = mlp_ultra_shallow(2, 3, hidden=8192).to(args.device)
    phi = MLP(2, 3).to(args.device)
    # phi = MLPWideAndDeep(2, 3).to(args.device)

    output_dict["uv"] = uv
    output_dict["x"] = x
    output_dict["transform"] = transform

    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    uv_optimizer = torch.optim.Adam([uv], lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters, return_transport_matrix=True)
    mse_loss = nn.MSELoss()

    # Cache correspondences to plot them later
    pi = None

    # Cache model with the lowest loss if --use-best is passed
    best_model = None
    best_loss = np.inf

    for epoch in range(args.local_epochs):
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        epoch_start_time = time.time()

        y = phi(uv)

        with torch.no_grad():
            if args.exact_emd:
                M = pairwise_distances(x.unsqueeze(0), y.unsqueeze(0)).squeeze().cpu().squeeze().numpy()
                p = ot.emd(np.ones(x.shape[0]), np.ones(x.shape[0]), M)
                p = torch.from_numpy(p.astype(np.float32)).to(args.device)
            else:
                _, p = sinkhorn_loss(x.unsqueeze(0), y.unsqueeze(0))
            pi = p.squeeze().max(0)[1]

        loss = mse_loss(x[pi].unsqueeze(0), y.unsqueeze(0))

        loss.backward()

        if args.use_best and loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(phi.state_dict())

        epoch_end_time = time.time()

        if epoch % args.print_every == 0:
            print("%d/%d: [Loss = %0.5f] [Time = %0.3f]" %
                  (epoch, args.local_epochs, loss.item(), epoch_end_time-epoch_start_time))

        optimizer.step()
        uv_optimizer.step()

    if args.use_best:
        phi.load_state_dict(best_model)

    output_dict["final_model"] = copy.deepcopy(phi.state_dict())

    torch.save(output_dict, args.output)

    if args.plot:
        plot_reconstruction(uv, x, transform, phi, pad=1.0)
        plot_correspondences(phi, uv, x, pi)


if __name__ == "__main__":
    main()
