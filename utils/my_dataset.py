import os.path as osp
import random
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm
from trimesh import Trimesh, load_mesh
import os
from utils.funcs import save_ply_explicit, get_edge_index
from concurrent.futures import ProcessPoolExecutor


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
        self.std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)
        data.x = (data.x - self.mean) / self.std
        data.y = (data.y - self.mean) / self.std
        return data


def load_mesh_file(file, train_dir):
    return load_mesh(osp.join(train_dir, file))


class MyDataset(InMemoryDataset):

    def __init__(self, config, dtype='train'):
        assert dtype in ['train', 'eval'], "Invalid dtype!"

        self.config = config
        self.root = config['dataset_dir']

        super(MyDataset, self).__init__(self.root)

        data_path = self.processed_paths[0]
        if dtype == 'eval':
            data_path = self.processed_paths[1]
        norm_path = self.processed_paths[2]
        edge_index_path = self.processed_paths[3]

        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)
        self.edge_index = torch.load(edge_index_path)

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'eval.pt', 'norm.pt', "edge_index.pt"]
        return processed_files

    def process(self):
        meshes = []
        train_data, eval_data = [], []
        train_vertices = []

        train_dir = osp.join(self.root, "train")
        files = os.listdir(train_dir)

        # for file in tqdm(files):
        #     mesh = load_mesh(osp.join(train_dir, file))
        #     meshes.append(mesh)

        with ProcessPoolExecutor(max_workers=8) as executor:  # 调整max_workers的数量以达到最佳性能
            futures = [executor.submit(load_mesh_file, file, train_dir) for file in files]
            for future in tqdm(futures):
                meshes.append(future.result())

        edge_index = get_edge_index(meshes[0].vertices, meshes[0].faces)

        # shuffle
        random.shuffle(meshes)
        count = int(0.8 * len(meshes))
        for i in range(len(meshes)):
            mesh_verts = torch.Tensor(meshes[i].vertices)
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)
            if i < count:
                train_data.append(data)
                train_vertices.append(mesh_verts)
            else:
                eval_data.append(data)

        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}

        # save template
        mesh = Trimesh(vertices=mean_train, faces=meshes[0].faces)
        save_ply_explicit(mesh, self.config['template_file'])

        print("transforming...")
        transform = Normalize(mean_train, std_train)
        train_data = [transform(x) for x in train_data]
        eval_data = [transform(x) for x in eval_data]

        # save
        print("saving...")
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(eval_data), self.processed_paths[1])
        torch.save(norm_dict, self.processed_paths[2])
        torch.save(edge_index, self.processed_paths[3])
        torch.save(norm_dict, osp.join(self.config['dataset_dir'], "norm.pt"))
