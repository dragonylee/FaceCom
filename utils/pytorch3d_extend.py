from pytorch3d.loss.point_mesh_distance import *
from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance
from pytorch3d.structures import Meshes, Pointclouds
import torch
from torch import Tensor
from trimesh import Trimesh


def point_mesh_face_distance_single_direction(
        meshes: Meshes,
        pcls: Pointclouds,
        min_triangle_area: float = 1e-6,
):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    return point_to_face


def distance_from_reference_mesh(points: Tensor, mesh_vertices: Tensor, mesh_faces: Tensor):
    """
        return distance^2 from mesh for every point in points
    """
    meshes = Meshes([mesh_vertices], [mesh_faces])
    pcs = Pointclouds([points])
    distances = point_mesh_face_distance_single_direction(meshes, pcs)
    return distances


def smoothness_loss(vertices: Tensor, faces: Tensor):
    meshes = Meshes([vertices], [faces])
    return mesh_laplacian_smoothing(meshes)
