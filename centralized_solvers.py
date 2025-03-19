import numpy as np
    
def pow_iter_update(cov_mat, evecs, local_iters, step_size):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1] # (k-1) * d matrix
    est_eval_sqs = np.multiply(prev_vs, prev_vs @ cov_mat).sum(axis=1)
    for local_idx in range(local_iters):
        reward = cov_mat @ v
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            penalty = prev_vs.T @ np.multiply(prev_vs @ v, est_eval_sqs)
        v = (reward - penalty) / np.linalg.norm(reward - penalty)
    return v


def hebb_update(cov_mat, evecs, local_iters, step_size):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1] # (k-1) * d matrix
    est_eval_sqs = np.multiply(prev_vs, prev_vs @ cov_mat).sum(axis=1)
    for _ in range(local_iters):
        reward = cov_mat @ v
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            penalty = prev_vs.T @ np.multiply(prev_vs @ v, est_eval_sqs)
        grad = reward - penalty
        v = (v + step_size * grad) / np.linalg.norm(v + step_size * grad)
    return v


def alpha_evec_update(cov_mat, evecs, local_iters, step_size):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1]
    prev_cov = prev_vs @ cov_mat
    est_eval_sqs = np.multiply(prev_vs, prev_cov).sum(axis=1) # dim (k-1) vector
    for _ in range(local_iters):
        reward = cov_mat @ v
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            coeffs = np.divide(prev_cov @ v, est_eval_sqs)
            penalty = prev_cov.T @ coeffs
        grad = reward - penalty
        # v += step_size * (grad - np.inner(grad, v) * v)
        if step_size > 0:
            v = (v + step_size * grad) / np.linalg.norm(v + step_size * grad)
        else:
            v = grad / np.linalg.norm(grad)
    return v


def mu_evec_update(cov_mat, evecs, local_iters, step_size):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1]
    for _ in range(local_iters):
        reward = cov_mat @ v
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            penalty = prev_vs.T @ (prev_vs @ (cov_mat @ v))
        grad = reward - penalty
        # v += step_size * (grad - np.inner(grad, v) * v)
        if step_size > 0:
            v = (v + step_size * grad) / np.linalg.norm(v + step_size * grad)
        else:
            v = grad / np.linalg.norm(grad)
    return v


def eigengame(cov_mat, num_evecs, global_com_rounds, local_iters, step_size_func, update='alpha'):
    cur_evecs = np.random.normal(size=(num_evecs, cov_mat.shape[1]))
    cur_evecs /= np.linalg.norm(cur_evecs, axis=1, keepdims=True)
    hist = [np.copy(cur_evecs)]
    update_func = alpha_evec_update
    if update == 'mu':
        update_func = mu_evec_update
    for g_idx in range(global_com_rounds):
        new_evecs = np.copy(cur_evecs)
        for evec_idx in range(num_evecs):
            if evec_idx > g_idx:
                pass
            else:
                new_evecs[evec_idx] = update_func(cov_mat,
                                                  cur_evecs[:evec_idx+1],
                                                  local_iters,
                                                  step_size_func(g_idx + 1))
        cur_evecs = new_evecs
        hist.append(np.copy(cur_evecs))
    return hist


def parallel_deflation(cov_mat, num_evecs, global_com_rounds, local_iters,
                       step_size_func, update='pw'):
    cur_evecs = np.random.normal(size=(num_evecs, cov_mat.shape[1]))
    cur_evecs /= np.linalg.norm(cur_evecs, axis=1, keepdims=True)
    hist = [np.copy(cur_evecs)]
    update_func = pow_iter_update
    if update == 'hebb':
        update_func = hebb_update
    for g_idx in range(global_com_rounds):
        new_evecs = np.copy(cur_evecs)
        for evec_idx in range(num_evecs):
            if evec_idx > g_idx:
                pass
            else:
                new_evecs[evec_idx] = update_func(cov_mat,
                                                  cur_evecs[:evec_idx+1],
                                                  local_iters,
                                                  step_size_func(g_idx + 1))
        cur_evecs = new_evecs
        hist.append(np.copy(cur_evecs))
    return hist