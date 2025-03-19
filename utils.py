import numpy as np

def generate_powerlaw_matrix(d, r, power, return_evals=False):
    left_svec, _ , _ = np.linalg.svd(np.random.normal(size=(d, r)),
                                     full_matrices=False)
    eig_vals = 1. / np.power(np.arange(r) + 1, power)
    if return_evals:
        return left_svec.T, left_svec @ np.diag(eig_vals) @ left_svec.T, eig_vals
    return left_svec.T, left_svec @ np.diag(eig_vals) @ left_svec.T


def generate_expdecay_matrix(d, r, base, return_evals=False):
    left_svec, _ , _ = np.linalg.svd(np.random.normal(size=(d, r)),
                                     full_matrices=False)
    eig_vals = 1. / np.power(base, np.arange(r) + 1)
    if return_evals:
        return left_svec.T, left_svec @ np.diag(eig_vals) @ left_svec.T, eig_vals
    return left_svec.T, left_svec @ np.diag(eig_vals) @ left_svec.T


def generate_data_from_cov(cov_sqrt, num_samples):
    # return np.random.multivariate_normal(np.zeros((cov.shape[0],)), cov,
    #                                      size=num_samples) \
    #     / np.sqrt(num_samples)
    raw_x = np.random.normal(size=(num_samples, cov_sqrt.shape[1]))
    return raw_x @ cov_sqrt.T


def random_batch(feature, batch_size):
    return feature[np.random.choice(feature.shape[0], size=batch_size)] \
        / np.sqrt(batch_size)


def sq_errors(true_evecs, est_evecs):
    """
    true_evecs: k * d matrix
    est_evecs: k * d matrix
    """
    return 2 - 2 * np.abs(np.multiply(true_evecs, est_evecs).sum(axis=1))


def compute_max_error(true_evecs, est_evecs):
    return np.sqrt(sq_errors(true_evecs, est_evecs).max())


def compute_avg_error(true_evecs, est_evecs):
    return np.sqrt(sq_errors(true_evecs, est_evecs).mean())


def generate_block_powerlaw_matrix(d, num_blocks, block_size):
    r = int(num_blocks * block_size)
    left_svec, _ , _ = np.linalg.svd(np.random.normal(size=(d, r)),
                                     full_matrices=False)
    eig_vals = 1. / np.array([[i+1 for _ in range(block_size)] for i in range(num_blocks)]).flatten()
    return left_svec.T, left_svec @ np.diag(eig_vals) @ left_svec.T


def subspace_overlap(subspace1, subspace2):
    return np.linalg.norm(subspace1 @ subspace2.T) ** 2


def compute_block_avg_acc(true_evecs, est_evecs, num_blocks, block_size):
    agg_acc = 0
    for block_idx in num_blocks:
        space1 = true_evecs[block_idx * block_size: (block_idx + 1) * block_size]
        space2 = est_evecs[block_idx * block_size: (block_idx + 1) * block_size]
        agg_acc += subspace_overlap(space1, space2)
    return np.sqrt(agg_acc / true_evecs.shape[0])