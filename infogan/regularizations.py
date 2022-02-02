import torch
import numpy as np

from infogan.codes import get_cont_code


def resample_cont_code(cont_code, n_cont_code):
    n_samples = cont_code.size(0)
    new_code = get_cont_code(n_samples, n_cont_code)
    fix_idx = np.random.randint(0, n_cont_code)
    new_code[:, fix_idx] = cont_code[:, fix_idx]
    return new_code, fix_idx

def resample_each_cont_code(cont_code, n_cont_code, epsilon=1e-8):
    codes = []
    for i in range(n_cont_code):
        code_i = cont_code.clone()
        code_i[:, i] += epsilon
        codes.append(code_i)
    return codes


def resample_each_disc_code(disc_code, n_disc_code):
    disc_code = disc_code.clone()
    ids = np.random.choice(len(n_disc_code), disc_code.size(0))
    for i, dim in enumerate(n_disc_code):
        from_idx = sum(n_disc_code[:i])
        original_val = torch.argmax(
            disc_code[ids == i, from_idx : from_idx + dim], 1
        )
        # TODO: might random a different disc
        val = (original_val + 1) % dim
        val = torch.nn.functional.one_hot(torch.tensor(val).long(), dim)
        disc_code[ids == i, from_idx : from_idx + dim] = val
    return disc_code


def soft_orthogonal_regularization_loss(sampled, resampled):
    """
    sampled: the prediction of code before resample,
              shape(traj_len, batch_size, 2)
    resampled: the prediction of code with resample,
              shape(traj_len, batch_size, 2)
    """
    batch_size = sampled.size(1)
    # original = torch.transpose(original, 0, 1).reshape(batch_size, 1, -1)
    # compared = torch.transpose(compared, 0, 1).reshape(batch_size, -1, 1)
    # torch.sum(original @ compared) == 0
    sampled = torch.transpose(sampled, 0, 1).reshape(batch_size, -1)
    resampled = torch.transpose(resampled, 0, 1).reshape(batch_size, -1)
    n = sampled.size(1)
    # https://proceedings.neurips.cc/paper/2018/file/bf424cb7b0dea050a42b9739eb261a3a-Paper.pdf
    # W^T@W, W(m, n) orthogonal iff m > n
    return torch.norm(sampled.T @ resampled - torch.eye(n).to(sampled)) / (
        n * n
    )


def cosine_similarity(a, b, epsilon=1e-8):
    inner_product_ab = torch.sum(a * b)
    inner_product_aa = torch.sum(a * a)
    inner_product_bb = torch.sum(b * b)
    return inner_product_ab / (
        torch.max(
            torch.sqrt(inner_product_aa) * torch.sqrt(inner_product_bb),
            torch.tensor(epsilon).to(a),
        )
    )


def cosine_similarity_points(a, b):
    return torch.mean(torch.abs(torch.nn.functional.cosine_similarity(a, b)))


def cosine_similarity_loss(sampled, resampled):
    """
    sampled: the prediction of code before resample,
              shape(traj_len, batch_size, 2)
    resampled: the prediction of code with resample,
              shape(traj_len, batch_size, 2)
    """

    if sampled.size(0) == 1:
        sampled = torch.squeeze(sampled)
        resampled = torch.squeeze(resampled)
        sampled = sampled / (torch.norm(sampled, dim=1).reshape(-1, 1) + 1e-22)
        resampled = resampled / (torch.norm(resampled, dim=1).reshape(-1, 1) + 1e-22)
        res = torch.mean(
            torch.abs(torch.nn.functional.cosine_similarity(sampled, resampled))
        )
        return res
    else:
        batch_size = sampled.size(1)
        sampled = torch.transpose(sampled, 0, 1)  # batch_size, traj_len, 2
        resampled = torch.transpose(resampled, 0, 1)
        # torch.sum(sampled[i] * resampled[i])
        # cosine_similarity(sampled[i], resampled[i])
        cos_batches = torch.tensor(
            [
                cosine_similarity_points(sampled[i], resampled[i])
                for i in range(batch_size)
            ]
        )
        return torch.mean(torch.abs(cos_batches))


def euclidean_distance_loss(sampled, resampled):
    batch_size = sampled.size(1)
    sampled = torch.transpose(sampled, 0, 1).reshape(batch_size, -1)
    resampled = torch.transpose(resampled, 0, 1).reshape(batch_size, -1)
    norm = torch.norm(sampled - resampled)
    return 1 / (norm + 1)
    # return torch.mean(
    #     torch.nn.functional.pairwise_distance(sampled, resampled)
    # )
