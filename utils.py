import numpy as np
import torch
from point_cloud_utils import sample_mesh_poisson_disk
import json
import os
import random


def meshgrid_face_indices(cols, rows=-1):
    rows = rows if rows > 0 else cols
    r, c = map(np.ravel, np.mgrid[0:rows-2:complex(rows-1), 0:cols-2:complex(cols-1)])
    base = r * cols + c
    f = np.array([
        np.concatenate([base, base+1]),
        np.concatenate([base + 1, base + cols + 1]),
        np.concatenate([base + cols, base + cols])], dtype=np.int).T

    # f = np.array([
    #     np.concatenate([base, base+1]),
    #     np.concatenate([base+1, base+w+1]),
    #     np.concatenate([base+w, base+w])], dtype=np.int).T

    return np.ascontiguousarray(f)


def meshgrid_vertices(w, urange=[0, 1], vrange=[0, 1]):
    g = np.mgrid[urange[0]:urange[1]:complex(w), vrange[0]:vrange[1]:complex(w)]
    v = np.vstack(map(np.ravel, g)).T
    return np.ascontiguousarray(v)


def meshgrid_from_lloyd_ts(model_ts, n, scale=1.0):
    model_ts_min = np.min(model_ts, axis=0)
    model_ts_max = np.max(model_ts, axis=0)
    urange = np.array([model_ts_min[0], model_ts_max[0]])
    vrange = np.array([model_ts_min[1], model_ts_max[1]])
    ctr_u = np.mean(urange)
    ctr_v = np.mean(vrange)
    urange = (urange - ctr_u) * scale + ctr_u
    vrange = (vrange - ctr_v) * scale + ctr_v
    return meshgrid_vertices(n, urange=urange, vrange=vrange)


def load_point_cloud_by_file_extension(file_name, compute_normals=False):
    import point_cloud_utils as pcu
    if file_name.endswith(".obj"):
        v, f, n = pcu.read_obj(file_name, dtype=np.float32)
    elif file_name.endswith(".off"):
        v, f, n = pcu.read_off(file_name, dtype=np.float32)
    elif file_name.endswith(".ply"):
        v, f, n, _ = pcu.read_ply(file_name, dtype=np.float32)
    elif file_name.endswith(".npts"):
        v, n = load_srb_range_scan(file_name)
        f = []
    else:
        raise ValueError("Invalid file extension must be one of .obj, .off, .ply, or .npts")

    if compute_normals and f.shape[0] > 0:
        n = pcu.per_vertex_normals(v, f)
    return v, n


def srb_to_ply(srb_filename, ply_filename):
    v, n = load_srb_range_scan(srb_filename)
    with open(ply_filename, "w") as plyf:
        plyf.write("ply\n")
        plyf.write("format ascii 1.0\n")
        plyf.write("element vertex %d\n" % v.shape[0])
        plyf.write("property float x\n")
        plyf.write("property float y\n")
        plyf.write("property float z\n")
        plyf.write("property float nx\n")
        plyf.write("property float ny\n")
        plyf.write("property float nz\n")
        plyf.write("end_header\n")
        for i in range(v.shape[0]):
            tup = tuple(v[i, :]) + tuple(n[i, :])
            plyf.write("%f %f %f %f %f %f\n" % tup)


def load_srb_range_scan(file_name):
    """
    Load a range scan point cloud from the Surface Reconstruction Benchmark dataset
    :param file_name: The file containing the point cloud
    :return: A pair (v, f) of vertices and normals both with shape [n, 3]
    """
    v = []
    n = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            x, y, z, nx, ny, nz = [float(p) for p in line.split()]
            v.append((x, y, z))
            n.append((nx, ny, nz))
    return np.array(v), np.array(n)


def json_to_object(json_file):
    """
    Load a JSON file to a python object where the field names of the JSON are members of the returned object.
    :param json_file: Path to a JSON file
    :return: A python object whose fields correspond to the JSON fields
    """
    if not os.path.exists(json_file):
        raise ValueError("Invalid JSON File %s" % json_file)
    with open(json_file, 'r') as f:
        json_args = json.load(f)

    if json_args is None:
        raise ValueError("Unable to read JSON File %s" % json_file)

    class Args(object):
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Args(**json_args)


def random_mesh_samples(v, f, n_samples=10 ** 4):
    """
    Generate `n_samples` point samples on the mesh described by (v, f)
    :param v: A [n, 3] array of vertex positions
    :param f: A [n, 3] array of indices into v
    :param n_samples: The number of samples to generate
    :return: (P, face_ids) where P is an array of shape [n_samples, 3] and face_ids[i] is the face which P[i, :] lies on
    """
    vec_cross = np.cross(v[f[:, 0], :] - v[f[:, 2], :],
                         v[f[:, 1], :] - v[f[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Contributed by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = v[f[sample_face_idx, 0], :]
    B = v[f[sample_face_idx, 1], :]
    C = v[f[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
        np.sqrt(r[:,0:1]) * r[:,1:] * C
    return P, sample_face_idx


def downsample_point_cloud(point_cloud, normals, target_num_pts, max_iters=4096, max_retries=5):
    """
    Given a point cloud of shape [n, d] where each row is a point, downsample it to have
    num_pts points which are as evenly separated as possible. This function uses binary search
    over the radius parameter for Poisson Disk Sampling to find a radius yielding exactly the
    desired number of points. If the point cloud cannot be downsampled, the function throws
    as RuntimeError
    :param point_cloud: An array of shape [n, d] where each row, point_cloud[i, :], is a point
    :param normals: Normals at each point in the point cloud
    :param target_num_pts: The target number of points to downsample to
    :param max_iters: The maximum number of binary search iterations to find a valid down-sampling
    :param max_retries: The maximum number of retries for binary search
    :return: A downsampled point cloud
    """

    V = point_cloud
    N = normals
    if V.shape[0] < target_num_pts:
        raise ValueError("Cannot downsample point cloud with %d points to a target "
                         "of %d points" % (V.shape[0], target_num_pts))
    if V.shape[0] == target_num_pts:
        return V

    bbox = np.max(V, axis=0) - np.min(V, axis=0)
    bbox_diag = np.linalg.norm(bbox)
    F = np.zeros(V.shape, dtype=np.int32)

    pts_range = [target_num_pts, target_num_pts]
    radius_range = [0.00001*bbox_diag, 0.3*bbox_diag]

    Pdown = np.zeros([0, 3])
    Ndown = np.zeros([0, 3])

    success = False
    for _ in range(max_retries):
        num_iters = 0
        while not (pts_range[0] <= Pdown.shape[0] <= pts_range[1]):
            mid = radius_range[0] + 0.5 * (radius_range[1] - radius_range[0])
            Pdown, Ndown = sample_mesh_poisson_disk(V, F, N, radius=mid)

            if Pdown.shape[0] < pts_range[0]:
                radius_range[1] = mid
            elif Pdown.shape[0] > pts_range[1]:
                radius_range[0] = mid
            num_iters += 1
            if num_iters > max_iters:
                break
        if num_iters > max_iters:
            success = False
            continue
        else:
            success = True
            break

    if not success:
        raise RuntimeError("Failed to downsample point cloud. Try again")

    return Pdown, Ndown


def surface_area(v, f):
    """
    Compute the surface area of the batch of triangle meshes defined by v and f
    :param v: A [b, nv, 3] tensor where each [i, :, :] are the vertices of a mesh
    :param f: A [b, nf, 3] tensor where each [i, :, :] are the triangle indices into v[i, :, :] of the mesh
    :return: A tensor of shape [b, 1] with the surface area of each mesh
    """
    idx = torch.arange(v.shape[0])
    tris = v[:, f, :][idx, idx, :, :]
    a = tris[:, :, 1, :] - tris[:, :, 0, :]
    b = tris[:, :, 2, :] - tris[:, :, 0, :]
    areas = torch.sum(torch.norm(torch.cross(a, b, dim=2), dim=2)/2.0, dim=1)
    return areas


def arclength(x):
    """
    Compute the arclength of a curve sampled at a sequence of points.
    :param x: A [b, n, d] tensor of minibaches of d-dimensional point sequences.
    :return: A tensor of shape [b] where each entry, i, is estimated arclength for the curve samples x[i, :, :]
    """
    v = x[:, 1:, :] - x[:, :-1, :]
    return torch.norm(v, dim=2).sum(1)


def curvature_2d(x):
    """
    Compute the discrete curvature for a sequence of points on a curve lying in a 2D embedding space
    :param x: A [b, n, 2] tensor where each [i, :, :] is a sequence of n points lying along some curve
    :return: A [b, n-2] tensor where each [i, j] is the curvature at the point x[i, j+1, :]
    """
    # x has shape [b, n, d]
    b = x.shape[0]
    n_x = x.shape[1]
    n_v = n_x - 2

    v = x[:, 1:, :] - x[:, :-1, :]                         # v_i = x_{i+1} - x_i
    v_norm = torch.norm(v, dim=2)
    v = v / v_norm.view(b, n_x-1, 1)                       # v_i = v_i / ||v_i||
    v1 = v[:, :-1, :].contiguous().view(b * n_v, 1, 2)
    v2 = v[:, 1:, :].contiguous().view(b * n_v, 1, 2)
    c_c = torch.bmm(v1, v2.transpose(1, 2)).view(b, n_v)   # \theta_i = <v_i, v_i+1>
    return torch.acos(torch.clamp(c_c, min=-1.0, max=1.0)) / v_norm[:, 1:]


def normals_curve_2d(x):
    """
    Compute approximated normals for a sequence of point samples along a curve in 2D.
    :param x: A tensor of shape [b, n, 2] where each x[i, :, :] is a sequence of n 2d point samples on a curve
    :return: A tensor of shape [b, n, 2] where each [i, j, :] is the estimated normal for point x[i, j, :]
    """
    b = x.shape[0]
    n_x = x.shape[1]

    n = torch.zeros(x.shape)
    n[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    n[:, -1, :] = (x[:, -1, :] - x[:, -2, :])
    n = n[:, :, [1, 0]]
    n[:, :, 0] = -n[:, :, 0]
    n = n / torch.norm(n, dim=2).view(b, n_x, 1)
    n[:, 1:, :] = 0.5*(n[:, 1:, :] + n[:, :-1, :])
    n = n / torch.norm(n, dim=2).view(b, n_x, 1)
    return n


def isnan(x):
    """
    Returns True if the tensor x contains NaNs
    :param x: A torch.Tensor
    :return: True if x contains NaNs
    """
    return bool(torch.max(torch.isnan(x)) > 0)


def seed_everything(seed):
    """
    Seed all the RNGs that are used by the programs in this repository
    :param seed: The random seed to use. If non-positive, a seed is chosen at random
    :return: The seed used for the RNGs
    """
    if seed < 0:
        seed = np.random.randint(np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed


class ValueOrRandomRange(object):
    """
    Holds either a numerical value of type int or float or a pair of those types, [a, b] where b > a.
    If a ValueOrRandomRange object is constructed with a scalar value, the value property returns that value.
    If a ValueOrRandomRange object is constructed with a pair, (a, b), then the value property
    returns a value uniformly sampled from the range [a, b)
    """
    def __init__(self, val):
        """
        Construct the container with the value or range specified by val (see class documentation).

        :param val: A float or int scalar or a pair of float or int scalars (a, b) where b > a
        """
        self._type = None
        if hasattr(val, '__len__') and not 1 <= len(val) <= 2:
            raise ValueError("If val is an array or list it must be of length 1 or 2")
        if hasattr(val, '__len__') and len(val) == 2 and val[0] >= val[1]:
            raise ValueError("If val is an array of the form (val0, val1), val0 must be less than val1")
        if hasattr(val, '__len__') and len(val) == 2 and type(val[0]) != type(val[1]):
            raise ValueError("If val is an array of the form (val0, val1), val0 and val1 must have the same type")

        if not hasattr(val, '__len__'):
            self._type = type(val)
            self._val = val
        elif hasattr(val, '__len__') and len(val) == 1:
            self._type = type(val[0])
            self._val = val[0]
        elif hasattr(val, '__len__') and len(val) == 2:
            self._type = type(val[0])
            self._val = val

        if self._type not in [float, int]:
            raise ValueError("Type of value must be an int or a float, got %s" % self._type)

    @property
    def value(self):
        if hasattr(self._val, '__len__'):
            if self._type == float:
                return self._val[0] + (self._val[1] - self._val[0]) * np.random.rand()
            elif self._type == int:
                return np.random.randint(self._val[0], self._val[1])
            else:
                assert False, "This should never happen!"
        else:
            return self._val

    @property
    def value_or_range(self):
        return self._val
