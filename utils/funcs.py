from typing import List
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
import torch
from .models import FMGenDecoder
from quad_mesh_simplify import simplify_mesh
from torch.nn import Parameter, ParameterList
from trimesh import Trimesh, load_mesh
import os
from os.path import join
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def save_ply_explicit(mesh, ply_file_path):
    vertices = mesh.vertices
    faces = mesh.faces
    with open(ply_file_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(vertices)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face {}\n'.format(len(faces)))
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')

        # Write vertices
        for vertex in vertices:
            f.write('{} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

        # Write faces
        for face in faces:
            f.write('{} '.format(len(face)))
            f.write(' '.join(str(idx) for idx in face))
            f.write('\n')


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_edge_index(vertices, faces):
    """

    Modified from https://github.com/pixelite1201/pytorch_coma/blob/master/mesh_operations.py

    :param vertices:
    :param faces:
    :return:
    """
    vpv = sp.csc_matrix((len(vertices), len(vertices)))

    for i in range(3):
        IS = faces[:, i]
        JS = faces[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    vpv = vpv.tocoo()
    return torch.tensor(np.vstack((vpv.row, vpv.col)), dtype=torch.int)


def get_transform_matrix(vertices1, vertices2, k=5):
    """
    Calculate the transformation matrix D such that vertices2 = D * vertices1, where for each vertex in vertices2,
    it is formed as a linear combination of coordinates of the k nearest vertices in vertices1.

    :param vertices1:
    :param vertices2:
    :param k:
    :return:
    """
    kdtree = cKDTree(vertices1)
    _, indices = kdtree.query(vertices2, k=k)
    D = np.zeros((vertices2.shape[0], vertices1.shape[0]))

    for i in range(vertices2.shape[0]):
        vs = vertices1[indices[i]]
        w = np.matmul(vertices2[i], np.linalg.pinv(vs))
        D[i, indices[i]] = w

    return D


def generate_VFDU(mesh: Trimesh, factors: List[float]):
    V, F, D, U = [], [], [], []

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces, dtype=np.uint32)
    V.append(vertices)
    F.append(faces)

    for factor in factors:
        # QEM简化网格
        new_vertices, new_faces = simplify_mesh(vertices, faces, vertices.shape[0] / factor)
        V.append(new_vertices)
        F.append(new_faces)

        d = get_transform_matrix(vertices, new_vertices, 9)
        u = get_transform_matrix(new_vertices, vertices, 9)
        D.append(d)
        U.append(u)

        vertices = new_vertices
        faces = new_faces

    return V, F, D, U


def generate_transform_matrices_trimesh(mesh: Trimesh, factors: List[float]):
    V, F, D, U = generate_VFDU(mesh, factors)

    A = []
    for i in range(len(F)):
        edge_index = get_edge_index(V[i], F[i])
        A.append(edge_index)

    V = [torch.tensor(v) for v in V]
    D = [torch.tensor(d, dtype=torch.float32) for d in D]
    U = [torch.tensor(u, dtype=torch.float32) for u in U]

    return V, A, D, U


def get_mesh_matrices(config):
    template_mesh = load_mesh(config["template"])
    _, A, D, U = generate_transform_matrices_trimesh(
        template_mesh, config["down_sampling_factors"]
    )
    pA = ParameterList([Parameter(a, requires_grad=False) for a in A])
    pD = ParameterList([Parameter(a, requires_grad=False) for a in D])
    pU = ParameterList([Parameter(a, requires_grad=False) for a in U])

    return pA, pD, pU


def load_generator(config):
    A, D, U = get_mesh_matrices(config)
    model = FMGenDecoder(config, A, U)
    model.load_state_dict(
        torch.load(os.path.join(config["checkpoint_dir"], "checkpoint_decoder.pt"), map_location='cuda')
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def load_norm(config):
    norm_dict = torch.load(join(config['data_dir'], "norm.pt"))
    return norm_dict['mean'], norm_dict['std']


def get_random_z(length, requires_grad=True, jitter=False):
    z = torch.randn(1, length)
    z = z / torch.sqrt(torch.sum(z ** 2))
    if jitter:
        p = torch.randn_like(z) / 10 * z
        z += p
    z.requires_grad = requires_grad
    return z


def spherical_regularization_loss(z, target_radius=1.0):
    z_norm = torch.norm(z, p=2, dim=1)
    deviation = z_norm - target_radius
    loss = torch.mean(deviation ** 2)
    return loss
