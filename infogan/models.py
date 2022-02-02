import torch
from torch import nn

from sgan.models import make_mlp, Encoder

# MARK: - Models


class QHead(nn.Module):
    def __init__(
        self,
        dims,
        n_disc_code,
        n_cont_code,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
    ):
        super().__init__()
        # [h_dim, mlp_dim, sum(n_disc_code)]
        self.disc_backbone = make_mlp(
            dims + [sum(n_disc_code)],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        # [h_dim, mlp_dim, n_cont_code*2]
        self.cont_backbone = make_mlp(
            dims + [n_cont_code * 2],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, x):
        disc_info = self.disc_backbone(x)
        cont_info = self.cont_backbone(x)
        return disc_info, cont_info


class DHead(nn.Module):
    def __init__(self, dims, activation="relu", batch_norm=True, dropout=0.0):
        super().__init__()
        # [h_dim, mlp_dim, 1]
        self.backbone = make_mlp(
            dims + [1],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class Recognizer(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        n_disc_code,
        n_cont_code,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        num_layers=1,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
    ):
        super(Recognizer, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.mlp = make_mlp(
            [self.h_dim*2, self.mlp_dim, n_cont_code*2],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        self.head = nn.Linear(n_cont_code*2, n_cont_code)

    def forward(self, traj_rel1, traj_rel2):
        x1 = self.encoder(traj_rel1)
        x1 = x1.squeeze()
        x2 = self.encoder(traj_rel2)
        x2 = x2.squeeze()

        x = torch.cat([x1, x2], axis=1)
        x = self.mlp(x)
        out = self.head(x)
        return out