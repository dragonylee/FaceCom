import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random
from config.config import read_config
from utils.my_dataset import MyDataset
from utils.models import FMGenModel
import argparse
from torch.nn import Conv1d, Parameter, ParameterList
from trimesh import Trimesh, load_mesh
from utils.funcs import get_mesh_matrices, spherical_regularization_loss
import warnings

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_model(config, load_state_dict):
    pA, pD, pU = get_mesh_matrices(config)
    model = FMGenModel(config, pA, pD, pU)
    if load_state_dict:
        model.encoder.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], 'checkpoint_encoder.pt')))
        model.decoder.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], 'checkpoint_decoder.pt')))

    return model


def train_epoch(model, train_loader, optimizer, device, size, epoch, lambda_reg=1.0):
    model.train()
    total_loss_l1 = 0
    total_loss_mse = 0
    total_loss_reg = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out, z = model(batch)

        loss_mse = F.mse_loss(out, batch.y)
        loss_l1 = F.l1_loss(out, batch.y)
        loss_reg = spherical_regularization_loss(z)

        total_loss_mse += batch.num_graphs * loss_mse.item()
        total_loss_l1 += batch.num_graphs * loss_l1.item()
        total_loss_reg += batch.num_graphs * loss_reg.item()

        loss = loss_l1 + loss_mse + lambda_reg * loss_reg

        loss.backward()
        optimizer.step()

    return total_loss_l1 / size, total_loss_mse / size, total_loss_reg / size


def test_epoch(model, test_loader, device, size):
    model.eval()
    total_loss_l1 = 0
    total_loss_mse = 0
    total_loss_reg = 0

    for batch in tqdm(test_loader):
        batch = batch.to(device)

        out, z = model(batch)

        loss_mse = F.mse_loss(out, batch.y)
        loss_l1 = F.l1_loss(out, batch.y)
        loss_reg = spherical_regularization_loss(z)

        total_loss_mse += batch.num_graphs * loss_mse.item()
        total_loss_l1 += batch.num_graphs * loss_l1.item()
        total_loss_reg += batch.num_graphs * loss_reg.item()

    return total_loss_l1 / size, total_loss_mse / size, total_loss_reg / size


def train(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])

    # #### dataset #####
    print("loading datasets...")
    dataset_train = MyDataset(config, 'train')
    dataset_test = MyDataset(config, 'eval')
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], pin_memory=True, persistent_workers=True)

    # #### model #####
    print("loading model...")
    model = load_model(config, False)
    model.to(device)

    # #### optimization #####
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # scheduler.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], 'scheduler.pt')))

    # #### train for epochs #####
    print("start training...")
    best_loss_item = float('inf')

    lambda_reg = config['lambda_reg']

    for epoch in range(scheduler.last_epoch + 1, config['epoch'] + 1):
        print("Epoch", epoch, "  lr:", scheduler.get_lr())

        loss_l1, loss_mse, loss_reg = train_epoch(model, train_loader, optimizer, device, len(dataset_train), epoch,
                                                  lambda_reg)
        print("Train    loss:    L1:", loss_l1, "MSE:", loss_mse, "REG:", loss_reg)

        loss_l1_test, loss_mse_test, loss_reg_test = test_epoch(model, test_loader, device, len(dataset_test))
        print("Test     loss:    L1:", loss_l1_test, "MSE:", loss_mse_test, "REG:", loss_reg_test)

        scheduler.step()

        if loss_l1_test + loss_mse_test + lambda_reg * loss_reg_test < best_loss_item:
            best_loss_item = loss_l1_test + loss_mse_test + lambda_reg * loss_reg_test
            torch.save(model.encoder.state_dict(), os.path.join(config['checkpoint_dir'], 'checkpoint_encoder.pt'))
            torch.save(model.decoder.state_dict(), os.path.join(config['checkpoint_dir'], 'checkpoint_decoder.pt'))
            torch.save(scheduler.state_dict(), os.path.join(config['checkpoint_dir'], 'scheduler.pt'))
            print("\nsave!\n\n")


def main():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    # args.config_file = "../config/test_config.cfg"

    config = read_config(args.config_file)
    train(config)


if __name__ == "__main__":
    main()
