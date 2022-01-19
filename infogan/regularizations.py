import torch
import numpy as np


def resample_each_cont_code(cont_code, n_cont_code):
    first_cont_code = cont_code.clone()
    second_cont_code = cont_code.clone()
    first_ids = np.random.choice(n_cont_code, cont_code.size(0))
    second_ids = (first_ids + 1) % n_cont_code

    for i in range(n_cont_code):
        first_cont_code[first_ids == i, i] += 1
        second_cont_code[second_ids == i, i] += 1
    return first_cont_code, second_cont_code


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
    return torch.norm(sampled.T @ resampled - torch.eye(n).to(sampled)) / (n * n)


def cosine_similarity_loss(sampled, resampled):
    """
    sampled: the prediction of code before resample,
              shape(traj_len, batch_size, 2)
    resampled: the prediction of code with resample,
              shape(traj_len, batch_size, 2)
    """
    batch_size = sampled.size(1)
    sampled = torch.transpose(sampled, 0, 1).reshape(batch_size, -1)
    resampled = torch.transpose(resampled, 0, 1).reshape(batch_size, -1)
    return torch.mean(
        torch.nn.functional.cosine_similarity(sampled, resampled)
    )

def euclidean_distance_loss(sampled, resampled):
    batch_size = sampled.size(1)
    sampled = torch.transpose(sampled, 0, 1).reshape(batch_size, -1)
    resampled = torch.transpose(resampled, 0, 1).reshape(batch_size, -1)
    norm = torch.norm(sampled - resampled)
    return 1 / (norm + 1)
    # return torch.mean(
    #     torch.nn.functional.pairwise_distance(sampled, resampled)
    # )