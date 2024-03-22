import torch
import torchvision.utils

from config.config import read_config
from trimesh import Trimesh, load_mesh
from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
import torch.nn.functional as F
import os
from os.path import join
import warnings
from tqdm import tqdm
import numpy as np
from queue import Queue

import math
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from .pytorch3d_extend import distance_from_reference_mesh, smoothness_loss
from trimesh.registration import icp
from scipy.spatial import cKDTree
import sys

from PIL import Image
import torchvision.transforms as transforms
from .render import render_d

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
sys.setrecursionlimit(30000)


def generate_face_sample(out_file, config, generator):
    generator.eval()
    device = generator.parameters().__next__().device
    z = get_random_z(generator.z_length, requires_grad=False)
    mean, std = load_norm(config)
    out = generator(z.to(device), 1).detach().cpu()
    out = out * std + mean
    template_mesh = load_mesh(config["template"])
    mesh = Trimesh(out, template_mesh.faces)
    save_ply_explicit(mesh, out_file)


def rigid_registration(in_mesh, config, verbose=True):
    if verbose:
        print("rigid registration...")

    mesh = in_mesh.copy()
    template_mesh = load_mesh(config["template"])

    mesh.vertices -= mesh.centroid
    T, _, _ = icp(mesh.vertices, template_mesh.vertices, max_iterations=50)
    mesh.apply_transform(T)

    return mesh


def fit(in_mesh, generator, config, device, max_iters=1000, loss_convergence=1e-6, lambda_reg=None,
        verbose=True, dis_percent=None):
    if verbose:
        sys.stdout.write("\rFitting...")
        sys.stdout.flush()

    mesh = in_mesh.copy()
    template_mesh = load_mesh(config["template"])

    generator.eval()

    target_pc = torch.tensor(mesh.vertices, dtype=torch.float).to(device)

    z = get_random_z(generator.z_length, requires_grad=True, jitter=True)

    mean, std = load_norm(config)
    mean = mean.to(device)
    std = std.to(device)
    faces = torch.tensor(template_mesh.faces).to(device)

    optimizer = torch.optim.Adam([z], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    if not lambda_reg:
        lambda_reg = config['lambda_reg']
    last_loss = math.inf
    iters = 0
    for i in range(max_iters):
        optimizer.zero_grad()

        out = generator(z.to(device), 1)
        out = out * std + mean

        loss_reg = spherical_regularization_loss(z)
        loss = loss_reg

        distance = torch.sqrt(distance_from_reference_mesh(target_pc, out, faces))
        if dis_percent:
            # 只取距离最小的一部分顶点
            distance, idx = torch.sort(distance)
            distance = distance[:int(dis_percent * len(distance))]
        loss_dfrm = torch.mean(distance)
        loss = loss_dfrm + lambda_reg * loss_reg

        if verbose:
            sys.stdout.write(
                "\rFitting...\tIter {}, loss_recon: {:.6f}, loss_reg: {:.6f}".format(i + 1,
                                                                                     loss_dfrm.item(),
                                                                                     loss_reg.item()))
            sys.stdout.flush()
        if math.fabs(last_loss - loss.item()) < loss_convergence:
            iters = i
            break

        last_loss = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    out = generator(z.to(device), 1)
    out = out * std + mean
    fit_mesh = Trimesh(out.detach().cpu(), template_mesh.faces)
    if verbose:
        print("")

    return fit_mesh


def post_processing(in_mesh_fit, in_mesh_faulty, device, laplacian=True, verbose=True):
    if verbose:
        print("post processing...")

    def get_color_mesh(mesh, idx, init_color=True, color=None):
        if color is None:
            color = [255, 0, 0, 255]
        color_mesh = mesh.copy()

        if init_color:
            color_array = np.zeros((mesh.vertices.shape[0], 4), dtype=np.uint8)  # RGBA颜色
            color_array[idx] = color
            color_mesh.visual.vertex_colors = color_array
        else:
            color_mesh.visual.vertex_colors[idx] = color
        return color_mesh

    def extract_connected_components(mesh: Trimesh, idx):
        visited = set()
        components = []

        def dfs(vertex, component):
            if vertex in visited:
                return
            visited.add(vertex)
            component.add(vertex)
            for neighbor in mesh.vertex_neighbors[vertex]:
                if neighbor in idx:
                    dfs(neighbor, component)

        for vertex in idx:
            if vertex not in visited:
                component = set()
                dfs(vertex, component)
                components.append(component)

        return components

    def expand_connected_component(mesh, component_, distance):
        expanded_component = set()
        component = component_.copy()

        for _ in range(distance):
            new_neighbors = set()
            for vertex in component:
                neighbors = mesh.vertex_neighbors[vertex]
                for neighbor in neighbors:
                    if neighbor not in component and neighbor not in expanded_component:
                        new_neighbors.add(neighbor)
            expanded_component.update(new_neighbors)
            component.update(new_neighbors)

        return expanded_component

    def special_point_refinement(mesh: Trimesh):
        vertices = mesh.vertices
        for i in tqdm(range(mesh.vertices.shape[0])):
            neighbor = mesh.vertex_neighbors[i]
            mean_x = np.mean(vertices[neighbor], axis=0)
            x = vertices[i]
            mean_distance = np.mean(np.linalg.norm(mesh.vertices[neighbor] - mean_x, axis=1))
            if np.linalg.norm(mean_x - x) > 0.5 * mean_distance:
                vertices[i] = mean_x
        return Trimesh(vertices, mesh.faces)

    def projection(source_mesh: Trimesh, largest_component_mask, target_mesh: Trimesh, max_iters=1000):
        x = torch.tensor(source_mesh.vertices, dtype=torch.float).to(device)
        normal_vectors = torch.tensor(source_mesh.vertex_normals, dtype=torch.float).to(device)
        ndf = torch.randn(source_mesh.vertices.shape[0]).detach().to(device)
        ndf.requires_grad = True

        optimizer = torch.optim.Adam([ndf], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        last_loss = math.inf
        for i in range(max_iters):
            optimizer.zero_grad()

            out = x + normal_vectors * torch.unsqueeze(ndf, 1)
            distance = distance_from_reference_mesh(out[~largest_component_mask],
                                                    torch.tensor(target_mesh.vertices, dtype=torch.float).to(device),
                                                    torch.tensor(target_mesh.faces).to(device))
            distance = torch.sqrt(distance)
            loss_dfrm = torch.mean(distance)

            loss_smoothness = smoothness_loss(out, torch.tensor(source_mesh.faces).to(device))
            # 投影时保持平滑，防止某些顶点投影到别的曲面上去

            # loss = loss_dfrm + 1 * loss_smoothness
            loss = loss_dfrm

            if verbose:
                sys.stdout.write("\rProjection... Iter {}, Loss: {}".format(i + 1, loss.item()))
                sys.stdout.flush()
            if i > 100 and math.fabs(last_loss - loss.item()) < 1e-6:
                break

            last_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        out = x + normal_vectors * torch.unsqueeze(ndf, 1)
        out = out.detach().cpu()

        # 还要把largest_component_mask对应顶点变为原来的顶点
        out[largest_component_mask] = torch.tensor(source_mesh.vertices, dtype=torch.float)[largest_component_mask]

        new_mesh = Trimesh(out, source_mesh.faces)
        if verbose:
            print("")

        return new_mesh

    def find_nearest_vertices(target_vertices, source_vertices, k=1):
        tree = cKDTree(source_vertices)
        distances, indices = tree.query(target_vertices, k=k)
        return indices, distances

    # 读取mesh
    fit_mesh = in_mesh_fit.copy()
    faulty_mesh = in_mesh_faulty.copy()

    # 1.阈值法识别fit_mesh中缺损部分（“修补”部分）
    distance = distance_from_reference_mesh(torch.tensor(fit_mesh.vertices, dtype=torch.float).to(device),
                                            torch.tensor(faulty_mesh.vertices, dtype=torch.float).to(device),
                                            torch.tensor(faulty_mesh.faces).to(device)).cpu().numpy()
    idx = np.where(distance > 4)[0]  # 阈值
    color_mesh = get_color_mesh(fit_mesh, idx)
    # color_mesh.export(join(out_path, "color_1.ply"))

    # 2.计算最大联通分量（缺损部分）以及扩展部分
    connected_components = extract_connected_components(fit_mesh, idx)
    if len(connected_components) == 0:
        return fit_mesh
    largest_component = max(connected_components, key=len)
    expanded_component = expand_connected_component(fit_mesh, largest_component, 2)
    color_mesh = get_color_mesh(fit_mesh, list(largest_component))
    color_mesh = get_color_mesh(color_mesh, list(expanded_component), False, [0, 255, 0, 255])
    # color_mesh.export(join(out_path, 'color_2.ply'))

    # 3.最优化法向位移场（投影）
    vertex_mask = np.zeros(len(fit_mesh.vertices), dtype=bool)
    vertex_mask[list(largest_component)] = True
    projection_mesh = projection(fit_mesh, vertex_mask, faulty_mesh)
    # projection_mesh.export(join(out_path, "projection.ply"))

    # 4.将最大联通分量的顶点逐一进行位移。 位移量：扩展部分中K个最近顶点[投影]时位移的均值
    vertices_expanded = fit_mesh.vertices[list(expanded_component)]
    normal_displacement = (projection_mesh.vertices - fit_mesh.vertices)[list(expanded_component)]
    indices, distances = find_nearest_vertices(fit_mesh.vertices, vertices_expanded, k=15)
    completion_mesh = projection_mesh.copy()
    for id in largest_component:
        mean_displacement = np.mean(normal_displacement[indices[id]], axis=0)
        completion_mesh.vertices[id] += mean_displacement

    def laplacian_smoothing(iterations=2, smoothing_factor=0.5):
        vertices_to_smooth = list(expanded_component)
        vertices_to_smooth.extend(list(largest_component))

        # 循环进行拉普拉斯平滑处理
        for _ in range(iterations):
            smoothed_vertices = []
            for vertex_index in vertices_to_smooth:
                vertex = completion_mesh.vertices[vertex_index]
                neighbors = completion_mesh.vertex_neighbors[vertex_index]
                neighbor_vertices = completion_mesh.vertices[neighbors]
                smoothed_vertex = vertex + smoothing_factor * np.mean(neighbor_vertices - vertex, axis=0)
                smoothed_vertices.append(smoothed_vertex)
            # 更新要平滑的顶点坐标
            for i, vertex_index in enumerate(vertices_to_smooth):
                completion_mesh.vertices[vertex_index] = smoothed_vertices[i]

    # 5.拉普拉斯平滑处理修复区域
    if laplacian:
        laplacian_smoothing()

    # 保存
    # completion_mesh.export(join(out_path, 'completion.ply'))

    # TODO: trimesh blender?
    #### 尝试把最大联通分量与缺损的组合在一起 ####
    # indices = list(largest_component)
    # new_vertices = completion_mesh.vertices[indices]
    # index_map = {index: i for i, index in enumerate(indices)}
    # new_faces = []
    # for face in completion_mesh.faces:
    #     new_face = [index_map[index] for index in face if index in index_map]
    #     if len(new_face) >= 3:
    #         new_faces.append(new_face)
    # new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    # # 缺损部分
    # new_mesh.export(join(out_path, 'fix_part.ply'))
    # # union
    # union_mesh = faulty_mesh.union(new_mesh)
    # union_mesh.export(join(out_path, "union.ply"))

    # refinement
    if verbose:
        print("refinement")
    for i in range(5):
        completion_mesh = special_point_refinement(completion_mesh)
    # completion_mesh.export(join(out_path, 'refinement.ply'))

    # done
    print("done!")
    return completion_mesh


def facial_mesh_completion(in_file, out_file, config, generator, lambda_reg=None, verbose=True, rr=False,
                           dis_percent=None):
    dir = os.path.dirname(in_file)
    device = generator.parameters().__next__().device

    mesh_in = load_mesh(in_file)

    if rr:
        mesh_in = rigid_registration(mesh_in, config, verbose=verbose)

    mesh_fit = fit(mesh_in, generator, config, device, lambda_reg=lambda_reg, verbose=verbose, loss_convergence=1e-7,
                   dis_percent=dis_percent)
    mesh_com = post_processing(mesh_fit, mesh_in, device, verbose=verbose)

    save_ply_explicit(mesh_com, out_file)
