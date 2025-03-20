import numpy as np
from tqdm import tqdm

def pow_iter_update(evecs, local_iters, step_size, get_random_batch):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1] # (k-1) * d matrix
    for _ in range(local_iters):
        batch_feature = get_random_batch()
        est_eval_sqs = np.linalg.norm(batch_feature @ prev_vs.T, axis=0) ** 2
        reward = batch_feature.T @ (batch_feature @ v)
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            penalty = prev_vs.T @ np.multiply(prev_vs @ v, est_eval_sqs)
        v = (reward - penalty) / np.linalg.norm(reward - penalty)
    return v


def hebb_update(evecs, local_iters, step_size, get_random_batch):
    v = np.copy(evecs[-1])
    prev_vs = evecs[:-1] # (k-1) * d matrix
    for _ in range(local_iters):
        batch_feature = get_random_batch()
        est_eval_sqs = np.linalg.norm(batch_feature @ prev_vs.T, axis=0) ** 2
        reward = batch_feature.T @ (batch_feature @ v)
        penalty = np.zeros_like(reward)
        if evecs.shape[0] > 1:
            penalty = prev_vs.T @ np.multiply(prev_vs @ v, est_eval_sqs)
        grad = reward - penalty
        v = (v + step_size * grad) / np.linalg.norm(v + step_size * grad)
    return v

def worker(evec_idx, cur_evecs, local_iters, step_size, get_random_batch, update_func):
    new_evec = update_func(cur_evecs[:evec_idx+1], local_iters, step_size, get_random_batch)
    return evec_idx, new_evec

def parallel_deflation_1(num_evecs, global_com_rounds, local_iters,
                       step_size_func, randomBatchGetter, update='pw', eval_func=None, num_workers=4, parallel_mode="pool"):
    if parallel_mode == "thread":
        import multiprocess.dummy as mp
    else:
        import multiprocess as mp
    cur_evecs = np.random.normal(size=(num_evecs, 50176))
    cur_evecs /= np.linalg.norm(cur_evecs, axis=1, keepdims=True)
    hist = [np.copy(cur_evecs)]
    update_func = pow_iter_update
    if update == 'hebb':
        update_func = hebb_update

    def rand_batch_func():
        return randomBatchGetter.getRandBatch()

    for g_idx in range(global_com_rounds // local_iters):
        print(f"Current round {g_idx}")

        
        new_evecs = np.copy(cur_evecs)
        with mp.Pool(processes=num_workers) as pool:
            results = []

            for evec_idx in range(num_evecs):
                if evec_idx > g_idx:
                    continue
                # generate a random batch
                # print("generating random batches!")
                # random_batches = [randomBatchGetter.getRandBatch() for _ in range(local_iters)]
                # print("done")
                results.append(pool.apply_async(worker, args=(evec_idx, cur_evecs, local_iters, step_size_func(g_idx + 1), rand_batch_func, update_func)))

            for res in results:
                evec_idx, new_evec = res.get()
                new_evecs[evec_idx] = new_evec

        cur_evecs = new_evecs
        if eval_func is not None:
            eval_result = eval_func(cur_evecs)
            print(f'{g_idx}: Current Evaluation Results is {eval_result:.3f}')
        hist.append(np.copy(cur_evecs))
    return hist


def evaluate_ImageNet(B, randomBatchGetter, est_evecs):
    K = est_evecs.shape[0]
    n = 1281167
    lambdas = np.zeros(K)
    for k in range(K):
        print(f"Computing for lambda {k}")
        v_k = est_evecs[k]
        # prev_evecs is a matrix of shape k - 1 * number of features
        # all the eigenvectors until the current one, exclusive, with each row representing an eigenvector; 
        prev_evecs = est_evecs[:k]
        # prev_evecs @ v_k is a vector [v1.T vk, v2.T vk, ..., v(k-1).T vk], of shape k-1 x 1
        x = prev_evecs.T @ prev_evecs @ v_k
        v_k_hat = (v_k - x) / np.linalg.norm(v_k - x)
        for j in tqdm(range(n // B - 1)):
            Y_j = randomBatchGetter.getRandBatch()
            # Y_j = Y[j * B : (j + 1) * B]
            # Y_j /= np.sqrt(B)
            lambdas[k] += np.linalg.norm(Y_j @ v_k_hat) ** 2
    # print(lambdas)
    return sum(lambdas[k - 1] / k for k in range(1, K + 1))
    

def parallel_deflation_save_only_last(num_evecs, global_com_rounds, local_iters,
                       step_size_func, randomBatchGetter, update='pw', eval_func=None, eval_B=None, eval_randomBatchGetter=None, num_workers=4, parallel_mode="pool"):
    if parallel_mode == "thread":
        import multiprocess.dummy as mp
    else:
        import multiprocess as mp
    cur_evecs = np.random.normal(size=(num_evecs, 50176))
    cur_evecs /= np.linalg.norm(cur_evecs, axis=1, keepdims=True)
    update_func = pow_iter_update
    if update == 'hebb':
        update_func = hebb_update
    
    def rand_batch_func():
        return randomBatchGetter.getRandBatch()
        
    for g_idx in range(global_com_rounds // local_iters):
        print(f"Current round {g_idx}")
        new_evecs = np.copy(cur_evecs)
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for evec_idx in range(num_evecs):
                if evec_idx > g_idx:
                    continue
                results.append(pool.apply_async(worker, args=(evec_idx, cur_evecs, local_iters, step_size_func(g_idx + 1), rand_batch_func, update_func)))
            for res in results:
                evec_idx, new_evec = res.get()
                new_evecs[evec_idx] = new_evec
        cur_evecs = new_evecs
        if eval_func is not None:
            eval_result = eval_func(eval_B, eval_randomBatchGetter, cur_evecs)
            print(f'{g_idx}: Current Evaluation Results is {eval_result:.3f}')
        # hist.append(np.copy(cur_evecs))
    return np.copy(cur_evecs)
    