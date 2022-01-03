import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.distributions.normal import Normal

from sgan.models import make_mlp, get_noise
from sgan.utils import relative_to_abs

# MARK: - Generate Code


def get_disc_code(n_samples, n_disc_code=[]):
    if n_disc_code is not None and len(n_disc_code) > 0:
        result = torch.cat(
            [
                torch.nn.functional.one_hot(
                    torch.randint(low=0, high=dim, size=(n_samples,)), dim
                )
                for dim in n_disc_code
            ],
            dim=1,
        )
        return result.cuda() if torch.cuda.is_available() else result
    return None


def get_cont_code(n_samples, n_cont_code=0):
    if n_cont_code > 0:
        result = torch.randn(n_samples, n_cont_code)
        return result.cuda() if torch.cuda.is_available() else result
    return None


def get_fixed_disc_code(n_samples, disc_code=[], n_disc_code=[]):
    if n_disc_code is not None and len(n_disc_code) > 0:
        result = torch.cat(
            [
                torch.nn.functional.one_hot(
                    torch.tensor(disc_code[i]).repeat(n_samples),
                    n_disc_code[i],
                )
                for i in range(len(n_disc_code))
            ],
            dim=1,
        )
        return result.cuda() if torch.cuda.is_available() else result
    return None


def get_fixed_cont_code(n_samples, cont_code=[], n_cont_code=0):
    if n_cont_code > 0:
        result = torch.tensor(cont_code).repeat(n_samples, 1).float()
        return result.cuda() if torch.cuda.is_available() else result


def get_latent_code(disc_code, cont_code):
    var = []
    if disc_code is not None:
        var.append(disc_code)
    if cont_code is not None:
        var.append(cont_code)
    if len(var) > 0:
        return torch.cat(var, dim=1)
    return None


def expand_code(code, seq_start_end):
    if code is None:
        return None
    return torch.cat(
        [
            code[idx].view(1, -1).repeat(ed.item() - st.item(), 1)
            for idx, (st, ed) in enumerate(seq_start_end)
        ],
        dim=0,
    )


# MARK: - Loss
# TODO: infogan pytorch donot add qloss to discriminator
# TODO: predict logvar


def logli(x_var, mean, stddev):
    """
    TODO: why use sum in logli
    """
    # return Normal(mean, stddev).log_prob(x_var).sum(axis=1)
    TINY = torch.tensor(1e-8).float()
    TINY = TINY.cuda() if torch.cuda.is_available() else TINY
    pi = 2 * torch.acos(torch.zeros(1))
    epsilon = (x_var - mean) / (stddev + TINY)
    return torch.sum(
        -0.5 * torch.log(2 * pi)
        - torch.log(stddev + TINY)
        - 0.5 * (epsilon ** 2),
        axis=1,
    )


def original_normal_dist_loss(x_var, mean, stddev):
    cont_log_q_c_given_x = logli(x_var, mean, stddev)
    cont_log_q_c = logli(x_var, mean=0, stddev=1)
    cont_cross_ent = torch.mean(-cont_log_q_c_given_x)
    cont_ent = torch.mean(-cont_log_q_c)
    cont_mi_est = cont_ent - cont_cross_ent
    return -cont_mi_est


def simplified_normal_dist_loss(x_var, mean, stddev):
    # maximize the prob
    return -Normal(mean, stddev).log_prob(x_var).mean()


def normal_dist_loss(x_var, mean, logvar):
    stddev = torch.sqrt(torch.exp(logvar))
    loss = simplified_normal_dist_loss(x_var, mean, stddev)
    return loss


def q_disc_loss_fn(disc_code, q_logits, n_disc_code):
    disc_loss = torch.tensor(0).float()
    disc_loss = disc_loss.cuda() if torch.cuda.is_available() else disc_loss
    if len(n_disc_code) > 0:
        st, ed = 0, 0
        for dim in n_disc_code:
            st, ed = ed, ed + dim
            cur_target = torch.argmax(disc_code[:, st:ed], 1)
            disc_loss += torch.nn.functional.cross_entropy(
                q_logits[:, st:ed], cur_target
            )
    return disc_loss


def q_cont_loss_fn(cont_code, q_cont, n_cont_code):
    cont_loss = torch.tensor(0).float()
    cont_loss = cont_loss.cuda() if torch.cuda.is_available() else cont_loss

    if n_cont_code > 0:
        q_mu, q_logvar = q_cont[:, :n_cont_code], q_cont[:, n_cont_code:]
        cont_loss = normal_dist_loss(cont_code, q_mu, q_logvar)
    return cont_loss


def q_loss_fn(q_info, code_tuple, n_disc_code, n_cont_code):
    disc_code, cont_code = code_tuple
    q_logits, q_cont = q_info
    disc_loss = q_disc_loss_fn(disc_code, q_logits, n_disc_code)
    cont_loss = q_cont_loss_fn(cont_code, q_cont, n_cont_code)
    return disc_loss, cont_loss


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


# MARK: - Visiual
# # ref: https://github.com/agrimgupta92/sgan/issues/5#issuecomment-414381686


def rmse(preds, targets):
    return np.sum((preds - targets) ** 2)


def set_fix_code(latent_code, n_disc_code, fix_code):
    latent_code = latent_code.clone()
    idx, val = fix_code
    if idx < len(n_disc_code):
        dim = n_disc_code[idx]
        val = torch.nn.functional.one_hot(torch.tensor(val).long(), dim)
        from_idx = sum(n_disc_code[:idx])
        latent_code[:, from_idx : from_idx + dim] = val
    else:
        from_idx = sum(n_disc_code) + idx - len(n_disc_code)
        latent_code[:, from_idx] = val
    return latent_code


def interpolate(
    batch,
    noise_mix_type,
    noise_dim,
    noise_type,
    n_disc_code,
    n_cont_code,
    generator,
    fix_code_idx,
    n_interpolation=8,
    fix_code_range=(-2, 2),
    n_views=5,
):

    (
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_gt_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ) = batch

    batch_size = obs_traj_rel.size(1)
    n_samples = (
        seq_start_end.size(0) if noise_mix_type == "global" else batch_size
    )

    user_noise = get_noise((1,) + noise_dim, noise_type)
    user_noise = user_noise.repeat(n_samples, 1)

    disc_code = get_disc_code(1, n_disc_code)
    cont_code = get_cont_code(1, n_cont_code)
    latent_code = get_latent_code(disc_code, cont_code)

    is_fix_disc = fix_code_idx < len(n_disc_code)

    interpolations = []
    length = n_disc_code[fix_code_idx] if is_fix_disc else n_interpolation
    for i in range(length):
        progress = i / (length - 1)
        from_val, to_val = (
            (0, n_disc_code[fix_code_idx] - 1)
            if is_fix_disc
            else fix_code_range
        )
        final_val = from_val * (1 - progress) + progress * to_val
        final_latent_code = set_fix_code(latent_code, n_disc_code, 
            (fix_code_idx, final_val))
        final_latent_code = final_latent_code.repeat(n_samples, 1)

        generator_out = generator(
            obs_traj,
            obs_traj_rel,
            seq_start_end,
            user_noise=user_noise,
            latent_code=final_latent_code,
        )

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        interpolations.append((
            obs_traj[:, :n_views],
            pred_traj_gt[:, :n_views],
            pred_traj_fake[:, :n_views]))
    
    return interpolations



def plot_interpolations(interpolations):
    """
    interpolations: list of tuples.
                    elem of tuple has size (traj_len, batch_size, 2)
    """
    fig = plt.figure(figsize=(32, 18))
    # fig.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, wspace=0, hspace=0)
    n_views = interpolations[0][0].size(1)
    # axs = gs.subplots(sharex=True, sharey=True)
    gs = fig.add_gridspec(n_views, len(interpolations), hspace=0, wspace=0)
    for i, axs in enumerate(gs.subplots(sharex="col", sharey="row")):
        for j, ax in enumerate(axs):
            traj_obs, traj_real, traj_fake = interpolations[j]
            view_traj_obs = traj_obs[:, i, :].cpu()
            view_traj_real = traj_real[:, i, :].cpu()
            view_traj_fake = traj_fake[:, i, :].cpu()
            ax.scatter(view_traj_obs[:, 0], view_traj_obs[:, 1], c='k', s=25)
            ax.scatter(view_traj_fake[:, 0], view_traj_fake[:, 1], c='r', s=25)
            ax.scatter(view_traj_real[:, 0], view_traj_real[:, 1], c='g', s=25)
            ax.label_outer()

    return fig
