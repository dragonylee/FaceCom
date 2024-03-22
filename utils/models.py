import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import FeaStConv
from torch_geometric.nn import BatchNorm
from torch_geometric.data.batch import Batch


class FMGenEncoder(torch.nn.Module):
    def __init__(self, config, A, D):
        super(FMGenEncoder, self).__init__()
        self.A = [torch.tensor(a, requires_grad=False) for a in A]
        self.D = [torch.tensor(a, requires_grad=False) for a in D]

        self.batch_norm = config['batch_norm']
        self.n_layers = config['n_layers']
        self.z_length = config['z_length']
        self.num_features_global = config['num_features_global']
        self.num_features_local = config['num_features_local']

        # conv layers
        self.encoder_convs_global = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_global[k], out_channels=self.num_features_global[k + 1])
            for k in range(self.n_layers)
        ])
        self.encoder_convs_local = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_local[k], out_channels=self.num_features_local[k + 1])
            for k in range(self.n_layers)
        ])

        # bn
        self.encoder_bns_local = torch.nn.ModuleList([
            BatchNorm(in_channels=self.num_features_local[k]) for k in range(self.n_layers + 1)
        ])
        self.encoder_bns_global = torch.nn.ModuleList([
            BatchNorm(in_channels=self.num_features_global[k]) for k in range(self.n_layers + 1)
        ])

        # linear layers
        self.encoder_lin = torch.nn.Linear(self.z_length + self.num_features_global[-1], self.z_length)
        self.encoder_local_lin = torch.nn.Linear(self.D[0].shape[1] * self.num_features_local[-1], self.z_length)

        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)

    def forward(self, x, batch_size):
        self.A = [a.to(x.device) for a in self.A]
        self.D = [d.to(x.device) for d in self.D]

        # 分叉
        x_global = x
        x_local = x

        """
            global
        """
        # x_global: [batch_size * D[0].shape[1], num_features_global[0]]
        for i in range(self.n_layers):
            # 卷积
            x_global = self.encoder_convs_global[i](x=x_global, edge_index=self.A[i])
            # 归一化
            if self.batch_norm:
                x_global = self.encoder_bns_global[i + 1](x_global)
            # 激活函数
            x_global = F.leaky_relu(x_global)
            # 下采样
            x_global = x_global.reshape(batch_size, -1, self.num_features_global[i + 1])
            y = torch.zeros(batch_size, self.D[i].shape[0], x_global.shape[2], device=x_global.device)
            for j in range(batch_size):
                y[j] = torch.mm(self.D[i], x_global[j])
            x_global = y
            x_global = x_global.reshape(-1, self.num_features_global[i + 1])
        # x_global: [batch_size * D[-1].shape[0], num_features_global[-1]]

        x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1])
        # x_global: [batch_size, D[-1].shape[0], num_features_global[-1]]

        # (mean pool & relu)
        x_global = torch.mean(x_global, dim=1)
        x_global = F.leaky_relu(x_global)
        # x_global: [batch_size, num_features_global[-1]]

        """
            local
        """
        # begin x_local: [batch_size * D[0].shape[1], num_features_local[0]]
        for i in range(self.n_layers):
            # 卷积
            x_local = self.encoder_convs_local[i](x=x_local, edge_index=self.A[0])
            # 归一化
            if self.batch_norm:
                x_local = self.encoder_bns_local[i + 1](x_local)
            # 激活函数
            x_local = F.leaky_relu(x_local)
        # x_local: [batch_size * D[0].shape[1], num_features_local[-1]]

        x_local = x_local.reshape(batch_size, -1)
        # x_local: [batch_size, D[0].shape[1] * num_features_local[0]]

        # (linear & relu)
        x_local = self.encoder_local_lin(x_local)
        x_local = F.leaky_relu(x_local)
        # x_local: [batch_size, z_length]

        """
            get z
        """
        z = torch.concat((x_global, x_local), dim=1)
        z = self.encoder_lin(z)

        return z


class FMGenDecoder(torch.nn.Module):
    def __init__(self, config, A, U):
        super(FMGenDecoder, self).__init__()
        self.A = [torch.tensor(a, requires_grad=False) for a in A]
        self.U = [torch.tensor(u, requires_grad=False) for u in U]

        self.batch_norm = config['batch_norm']
        self.n_layers = config['n_layers']
        self.z_length = config['z_length']
        self.num_features_global = config['num_features_global']
        self.num_features_local = config['num_features_local']

        # conv layers
        self.decoder_convs_global = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_global[-1 - k], out_channels=self.num_features_global[-2 - k])
            for k in range(self.n_layers)
        ])
        self.decoder_convs_local = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_local[-1 - k], out_channels=self.num_features_local[-2 - k])
            for k in range(self.n_layers)
        ])

        # bn
        self.decoder_bns_local = torch.nn.ModuleList([
            BatchNorm(in_channels=self.num_features_local[-1 - k]) for k in range(self.n_layers + 1)
        ])
        self.decoder_bns_global = torch.nn.ModuleList([
            BatchNorm(in_channels=self.num_features_global[-1 - k]) for k in range(self.n_layers + 1)
        ])

        # linear layers
        self.decoder_lin = torch.nn.Linear(self.z_length, self.z_length + self.num_features_global[-1])
        self.decoder_local_lin = torch.nn.Linear(self.z_length, self.num_features_local[-1] * self.U[0].shape[0])

        self.reset_parameter()

        # merge ratio
        self.global_ratio = 0.01
        self.local_ratio = 1 - self.global_ratio

    def reset_parameter(self):
        torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)

    def forward(self, z, batch_size):
        self.A = [a.to(z.device) for a in self.A]
        self.U = [u.to(z.device) for u in self.U]

        # decoder linear & 分叉
        x = self.decoder_lin(z)
        x_global = x[:, :self.num_features_global[-1]]
        x_local = x[:, self.num_features_global[-1]:]

        """
            global
        """
        # x_global: [batch_size, num_features_global[-1]]
        x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
        # x_global: [batch_size, U[-1].shape[1], num_features_global[-1]]

        for i in range(self.n_layers):
            # 上采样
            x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
            y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
            for j in range(batch_size):
                y[j] = torch.mm(self.U[-1 - i], x_global[j])
            x_global = y
            x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
            # 卷积
            x_global = self.decoder_convs_global[i](x=x_global, edge_index=self.A[-2 - i])
            if i < self.n_layers - 1:
                # 归一化
                if self.batch_norm:
                    x_global = self.decoder_bns_global[i + 1](x_global)
                # 激活函数
                x_global = F.leaky_relu(x_global)
        # x_global: [batch_size, U[0].shape[0], num_features_global[0]]
        x_global = x_global.reshape(-1, self.num_features_global[0])
        # x_global: [batch_size * U[0].shape[0], num_features_global[0]]

        """
            local
        """
        # x_local: [batch_size, z_length]
        x_local = self.decoder_local_lin(x_local)
        # x_local: [batch_size, num_features_local[-1] * U[0].shape[0]]
        x_local = x_local.reshape(-1, self.num_features_local[-1])
        # x_local: [batch_size * U[0].shape[0], num_features_local[-1]]

        for i in range(self.n_layers):
            # 卷积
            x_local = self.decoder_convs_local[i](x=x_local, edge_index=self.A[0])
            if i < self.n_layers - 1:
                # 归一化
                if self.batch_norm:
                    x_local = self.decoder_bns_local[i + 1](x_local)
                # 激活函数
                x_local = F.leaky_relu(x_local)
        # x_local: [batch_size * U[0].shape[0], num_features_local[0]]

        """
            merge
        """
        x = self.global_ratio * x_global + self.local_ratio * x_local

        return x


class FMGenModel(torch.nn.Module):
    def __init__(self, config, A, D, U):
        super(FMGenModel, self).__init__()

        self.encoder = FMGenEncoder(config, A, D)
        self.decoder = FMGenDecoder(config, A, U)

    def forward(self, batch: Batch):
        batch_size = batch.num_graphs

        z = self.encoder(batch.x, batch_size)
        x = self.decoder(z, batch_size)

        return x, z
