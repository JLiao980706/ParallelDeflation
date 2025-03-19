import numpy as np


def hebb_update(data_gen, evecs, local_iters, step_size, batch_size):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1] # (k-1) * d matrix
    for _ in range(local_iters):
        batch_feature = data_gen(batch_size)
        est_eval_sqs = np.linalg.norm(batch_feature @ prev_vs.T, axis=0) ** 2
        reward = batch_feature.T @ (batch_feature @ v)
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            penalty = prev_vs.T @ np.multiply(prev_vs @ v, est_eval_sqs)
        grad = reward - penalty
        v = (v + step_size * grad) / np.linalg.norm(v + step_size * grad)
    return v


def alpha_evec_update(data_gen, evecs, local_iters, step_size, batch_size):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1]
    for _ in range(local_iters):
        batch_feature = data_gen(batch_size)
        X_prev_v = batch_feature @ prev_vs.T # n x k
        X_cur_v = batch_feature @ v # n x 1
        reward = batch_feature.T @ X_cur_v
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            coeffs = np.divide(X_prev_v.T @ X_cur_v,
                               np.linalg.norm(X_prev_v, axis=0) ** 2) # k
            penalty = batch_feature.T @ (X_prev_v @ coeffs)
        grad = reward - penalty
        v = (v + step_size * grad) / np.linalg.norm(v + step_size * grad)
    return v


def mu_evec_update(data_gen, evecs, local_iters, step_size, batch_size):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1] # k x d
    for _ in range(local_iters):
        batch_feature = data_gen(batch_size)
        X_prev_v = batch_feature @ prev_vs.T # n x k
        X_cur_v = batch_feature @ v # n x 1
        reward = batch_feature.T @ X_cur_v
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            penalty = prev_vs.T @ (prev_vs @ reward)
        grad = reward - penalty
        v = (v + step_size * grad) / np.linalg.norm(v + step_size * grad)
    return v


def parallel_deflation(data_gen, num_evecs, global_com_rounds, local_iters,
                       step_size_func, batch_size, eval_func=None):
    input_dim = data_gen(batch_size).shape[1]
    cur_evecs = np.random.normal(size=(num_evecs, input_dim))
    cur_evecs /= np.linalg.norm(cur_evecs, axis=1, keepdims=True)
    hist = [np.copy(cur_evecs)]
    for g_idx in range(global_com_rounds):
        new_evecs = np.copy(cur_evecs)
        for evec_idx in range(num_evecs):
            if evec_idx > g_idx:
                pass
            else:
                new_evecs[evec_idx] = hebb_update(data_gen,
                                                  cur_evecs[:evec_idx+1],
                                                  local_iters,
                                                  step_size_func(g_idx + 1),
                                                  batch_size)
        cur_evecs = new_evecs
        if eval_func is not None:
            eval_result = eval_func(cur_evecs)
            print(f'Current Evaluation Results is {eval_result:.3f}')
        hist.append(np.copy(cur_evecs))
    return hist


def eigengame(data_gen, num_evecs, global_com_rounds, local_iters,
              step_size_func, batch_size, update='alpha', eval_func=None):
    input_dim = data_gen(batch_size).shape[1]
    cur_evecs = np.random.normal(size=(num_evecs, input_dim))
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
                new_evecs[evec_idx] = update_func(data_gen,
                                                  cur_evecs[:evec_idx+1],
                                                  local_iters,
                                                  step_size_func(g_idx + 1),
                                                  batch_size)
        cur_evecs = new_evecs
        if eval_func is not None:
            eval_result = eval_func(cur_evecs)
            print(f'Current Evaluation Results is {eval_result:.3f}')
        hist.append(np.copy(cur_evecs))
    return hist
