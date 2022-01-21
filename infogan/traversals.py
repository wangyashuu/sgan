import torch
import matplotlib.pyplot as plt

from sgan.models import get_noise
from sgan.utils import relative_to_abs

from infogan.codes import get_disc_code, get_cont_code, get_latent_code

# MARK: - Latent Traversal
# Visiual ref: https://github.com/agrimgupta92/sgan/issues/5#issuecomment-414381686


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
    n_disc_code,
    n_cont_code,
    generator,
    user_noise,
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
        final_latent_code = set_fix_code(
            latent_code, n_disc_code, (fix_code_idx, final_val)
        )
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

        interpolations.append(
            (
                obs_traj[:, :n_views],
                pred_traj_gt[:, :n_views],
                pred_traj_fake[:, :n_views],
            )
        )

    return interpolations


def plot_interpolations(interpolations):
    """
    interpolations: list of tuples.
                    elem of tuple has size (traj_len, batch_size, 2)
    """
    n_views = interpolations[0][0].size(1)
    n_interpolations = len(interpolations)
    fig = plt.figure(figsize=(n_interpolations*3.2, n_views * 3.2))
    fig.subplots_adjust(
        bottom=0.05, top=0.95, left=0.05, right=0.95, wspace=0, hspace=0
    )
    # axs = gs.subplots(sharex=True, sharey=True)
    gs = fig.add_gridspec(n_views, n_interpolations, hspace=0, wspace=0)
    for i, axs in enumerate(gs.subplots(sharex="col", sharey="row")):
        for j, ax in enumerate(axs):
            traj_obs, traj_real, traj_fake = interpolations[j]
            view_traj_obs = traj_obs[:, i, :].cpu()
            view_traj_real = traj_real[:, i, :].cpu()
            view_traj_fake = traj_fake[:, i, :].cpu()
            ax.scatter(view_traj_obs[:, 0], view_traj_obs[:, 1], c="k", s=25)
            ax.scatter(view_traj_fake[:, 0], view_traj_fake[:, 1], c="r", s=25)
            ax.scatter(view_traj_real[:, 0], view_traj_real[:, 1], c="g", s=25)
            ax.label_outer()

    return fig
