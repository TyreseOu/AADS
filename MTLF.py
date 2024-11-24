import random
from scipy.sparse import issparse
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cholesky

"""

    Author: Teng Ouyang
    Updated date: 2024-11
    Description: Mahalanobis distance-based optimization
    GitHub: https://github.com/TyreseOu/AADS

"""


def compmedDist(X):
    size1 = X.shape[0]
    Xmed = X
    G = np.sum((Xmed * Xmed), axis=1)
    Q = np.tile(G.reshape(-1, 1), (1, size1))
    R = np.tile(G.reshape(size1, 1), (1, size1))
    dists = Q + R - 2 * Xmed.dot(Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(size1 ** 2, 1)
    sigma = np.sqrt(0.5 * np.median(dists[dists > 0]))
    return sigma


def kernel_Gaussian(x, c, sigma):
    d, nx = x.shape
    _, nc = c.shape
    x2 = np.sum(x ** 2, axis=0)
    c2 = np.sum(c ** 2, axis=0)

    distance2 = np.tile(c2, (nx, 1)) + np.tile(x2.reshape(nx, 1), (1, nc)) - 2 * x.T.dot(c)
    return np.exp(-distance2 / (2 * sigma ** 2))


def ComputeDistanceExtremes(X, a, b):
    n = X.shape[0]
    num_trials = min(100, n * (n - 1) // 2)
    dists = np.zeros((num_trials, 1))

    for i in range(num_trials):
        j1 = random.randint(0, n - 1)
        j2 = random.randint(0, n - 1)
        dists[i] = np.sum((X[j1, :] - X[j2, :]) ** 2)

    f, c = np.histogram(dists, bins=100)
    cumulative_frequencies = np.cumsum(f)

    l_index = np.searchsorted(cumulative_frequencies, a * num_trials / 100, side='right')
    u_index = np.searchsorted(cumulative_frequencies, b * num_trials / 100, side='right')

    l = np.sqrt(dists[l_index][0])
    u = np.sqrt(dists[u_index][0])

    return l, u


def GetConstraints(y, num_constraints, l, u):
    m = len(y)
    C = np.zeros((num_constraints, 4))

    unique_labels = np.unique(y)
    num_unique_labels = len(unique_labels)

    if num_unique_labels == 1 or m < 2:
        return np.array([])

    for k in range(num_constraints):
        i, j = np.random.choice(m, size=2, replace=False)
        if y[i] == y[j]:
            C[k, :] = [i, j, 1, l]
        else:
            C[k, :] = [i, j, -1, u]

    index1 = np.where(C[:, 2] == 1)[0]
    index2 = np.where(C[:, 2] == -1)[0]
    if len(index2) > 0:
        C[index2, 2] = -len(index1) / len(index2)
    else:
        print("Unbalance Processing")

    return C


def PCA_reduce(X, retain_dimensions):
    U, _, _ = np.linalg.svd(np.cov(X.T))
    reduced_X = X.dot(U[:, :retain_dimensions])
    return reduced_X


def RuLSIF(x_nu, x_de, x_re=None, alpha=0.5, sigma_list=None, lambda_list=None, b=100, fold=5):
    np.random.seed(1)

    d, n_de = x_de.shape
    d_nu, n_nu = x_nu.shape

    is_disp = True if x_re is not None else False

    if alpha is None:
        alpha = 0.5

    if sigma_list is None:
        x = np.concatenate([x_nu, x_de], axis=1)
        med = compmedDist(x.T)
        sigma_list = med * np.array([0.6, 0.8, 1, 1.2, 1.4])
    elif np.any(sigma_list <= 0):
        raise ValueError("Gaussian width must be positive")

    if lambda_list is None or len(lambda_list) == 0:
        lambda_list = 10.0 ** np.array([-3, -2, -1, 0, 1])
    elif np.any(lambda_list < 0):
        raise ValueError("Regularization parameter must be non-negative")

    b = min(b, n_nu)
    n_min = min(n_de, n_nu)

    score_cv = np.zeros((len(sigma_list), len(lambda_list)))

    if len(sigma_list) == 1 and len(lambda_list) == 1:
        sigma_chosen = sigma_list[0]
        lambda_chosen = lambda_list[0]
    else:
        if fold != 0:
            cv_index_nu = np.random.permutation(n_nu)
            cv_split_nu = np.floor(np.arange(n_nu) * fold / n_nu).astype(int) + 1
            cv_index_de = np.random.permutation(n_de)
            cv_split_de = np.floor(np.arange(n_de) * fold / n_de).astype(int) + 1

        for sigma_index in range(len(sigma_list)):
            sigma = sigma_list[sigma_index]
            x_ce = x_nu[:, np.random.permutation(n_nu)[:b]]
            K_de = kernel_Gaussian(x_de, x_ce, sigma).T
            K_nu = kernel_Gaussian(x_nu, x_ce, sigma).T

            score_tmp = np.zeros((fold, len(lambda_list)))
            for k in range(1, fold + 1):
                Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]]
                Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]]

                Ktmp = alpha / Ktmp2.shape[1] * Ktmp2 @ Ktmp2.T + (1 - alpha) / Ktmp1.shape[1] * Ktmp1 @ Ktmp1.T
                mKtmp = np.mean(K_nu[:, cv_index_nu[cv_split_nu != k]], axis=1)

                for lambda_index in range(len(lambda_list)):
                    lambda_val = lambda_list[lambda_index]
                    thetat_cv = mylinsolve(Ktmp + lambda_val * np.eye(b), mKtmp)
                    thetah_cv = thetat_cv
                    score_tmp[k - 1, lambda_index] = alpha * np.mean(
                        (K_nu[:, cv_index_nu[cv_split_nu == k]].T @ thetah_cv) ** 2) / 2 \
                                                     + (1 - alpha) * np.mean(
                        (K_de[:, cv_index_de[cv_split_de == k]].T @ thetah_cv) ** 2) / 2 \
                                                     - np.mean(K_nu[:, cv_index_nu[cv_split_nu == k]].T @ thetah_cv)

            score_cv[sigma_index, :] = np.mean(score_tmp, axis=0)

        score_cv_tmp = np.min(score_cv, axis=1)
        sigma_chosen_index = np.argmin(score_cv_tmp)
        sigma_chosen = sigma_list[sigma_chosen_index]

        score_cv_tmp = np.min(score_cv, axis=0)
        lambda_chosen_index = np.argmin(score_cv_tmp)
        lambda_chosen = lambda_list[lambda_chosen_index]

    K_de = kernel_Gaussian(x_de, x_nu[:, :b], sigma_chosen).T
    K_nu = kernel_Gaussian(x_nu, x_nu[:, :b], sigma_chosen).T

    if is_disp:
        K_re = kernel_Gaussian(x_re, x_nu[:, :b], sigma_chosen).T

    Ktmp = alpha / K_nu.shape[1] * K_nu @ K_nu.T + (1 - alpha) / K_de.shape[1] * K_de @ K_de.T
    mKtmp = np.mean(K_nu, axis=1)
    thetat = mylinsolve(Ktmp + lambda_chosen * np.eye(b), mKtmp)
    thetah = thetat

    if is_disp:
        nu_re = K_re.T @ thetah
    else:
        nu_re = None

    wh_x_de = (K_de.T @ thetah).flatten()
    wh_x_nu = (K_nu.T @ thetah).flatten()

    if is_disp:
        wh_x_re = (K_re.T @ thetah).flatten()
    else:
        wh_x_re = 0

    PE = np.mean(wh_x_nu) - 1 / 2 * (alpha * np.mean(wh_x_nu ** 2) + (1 - alpha) * np.mean(wh_x_de ** 2)) - 1 / 2

    wh_x_de = np.maximum(0, wh_x_de)
    wh_x_re = np.maximum(0, wh_x_re)

    return PE, wh_x_de, wh_x_re


def pdf_Gaussian(x, mu, sigma):
    d, nx = x.shape
    tmp = (x - np.tile(mu, (1, nx))) / np.tile(sigma, (1, nx)) / np.sqrt(2)
    px = (2 * np.pi) ** (-d / 2) / np.prod(sigma) * np.exp(-np.sum(tmp ** 2, axis=0))
    return px


def mylinsolve(A, b):
    sflag = issparse(A)

    if sflag:
        R = cholesky(A.todense(), lower=True, overwrite_a=True)
        x = np.linalg.solve(R.T, np.linalg.solve(R, b))
    else:
        R = cholesky(A, lower=True, overwrite_a=True)
        x = np.linalg.solve(R.T, np.linalg.solve(R, b))

    return x


def KNN(y, X, M, k, Xt):
    add1 = 0
    if np.min(y) == 0:
        y = y + 1
        add1 = 1

    n, m = X.shape
    nt, _ = Xt.shape

    K = np.dot(X, M).dot(M.T).dot(Xt.T)

    l = np.zeros(n)
    lt = np.zeros(nt)

    for i in range(n):
        l[i] = np.dot(X[i, :], M).dot(M.T).dot(X[i, :])

    for i in range(nt):
        lt[i] = np.dot(Xt[i, :], M).dot(M.T).dot(Xt[i, :])

    return K


def demo(src_features, tgt_train_features, tgt_test_features, tgt_train_y, tgt_test_y, src_y):
    src_n = src_features.shape[0]
    tar_ntra = tgt_train_features.shape[0]
    tar_ntes = tgt_test_features.shape[0]
    dim = src_features.shape[1]
    P = [1, 100, 11, 1, 100, 1, 1e-6, 1e-7]
    param = initParam(P)
    # print("run MTLF")
    A, wt = MTLF(param, src_features, tgt_train_features, tgt_test_features, tgt_train_y, tgt_test_y, src_y)

    return wt


def initParam(P):
    P = [1, 100, 11, 1, 100, 1, 1e-6, 1e-7]
    param = {}
    param['k'] = P[0]
    param['num_constraints'] = P[1]
    param['dim'] = P[2]
    param['sigma'] = P[3]
    param['lamda'] = P[4]
    param['beta'] = P[5]
    param['gamma'] = P[6]
    param['gammaW'] = P[7]
    param['a'] = 5
    param['b'] = 95
    param['epsilon'] = 1e-7
    return param


def MTLF(param, src_features, tgt_train_features, tgt_test_features, tgt_train_y, tgt_test_y, src_y):
    a = param['a']
    b = param['b']
    num_constraints = param['num_constraints']
    k = param['k']
    dim = param['dim']

    source_data = src_features.cpu().detach().numpy()
    target_data = tgt_train_features.cpu().detach().numpy()

    X = np.vstack((source_data, target_data))

    tgt_train_y = tgt_train_y.cpu().detach().numpy()
    tgt_test_y = tgt_test_y.cpu().detach().numpy()
    source_labels = src_y.cpu().detach().numpy().flatten().reshape(-1, 1)
    target_labels = tgt_train_y.flatten().reshape(-1, 1)
    y = np.vstack((source_labels, target_labels))

    Xtest = tgt_test_features.cpu().detach().numpy()

    ntra_s_data = source_data.shape
    ntra_t_data = target_data.shape
    ntra_s = ntra_s_data[0]
    ntra_t = ntra_t_data[0]
    x_source = X[:ntra_s, :]
    x_target = np.vstack((X[ntra_s:ntra_s + ntra_t, :], Xtest))

    _, wh_x_source, _ = RuLSIF(x_target.T, x_source.T)
    wh_x_target = np.ones(Xtest.shape[0])
    wh_x_source = np.hstack((wh_x_source, wh_x_target))

    l, u = ComputeDistanceExtremes(X, a, b)

    C = GetConstraints(y, num_constraints, l, u)

    if len(C) == 0:
        wt = np.zeros(X.shape[0])
        K = KNN(y, X, np.eye(X.shape[1]), k, Xtest)
        return K, wt

    Xci = X[C[:, 0].astype(int), :]
    Xcj = X[C[:, 1].astype(int), :]
    d = X.shape[1]
    p = num_constraints
    w0 = wh_x_source
    sd_tra = X[:ntra_s, :]
    td_tra = X[ntra_s:ntra_s + ntra_t, :]
    A, result = optimization(C, w0, Xci, Xcj, param, sd_tra, td_tra)
    wt = result['wt']
    K = KNN(y, X, A, k, Xtest)
    return K, wt


def optimization(C, w0, Xci, Xcj, param, sd_tra, td_tra):
    epsilon = param['epsilon']
    sigma = param['sigma']
    lamda = param['lamda']
    beta = param['beta']
    gamma = param['gamma']
    gammaW = param['gammaW'] / sigma

    A0 = np.eye(Xci.shape[1])
    E = A0
    At = A0

    ns = sd_tra.shape[0]
    nt = td_tra.shape[0]
    e = np.zeros(ns + nt)
    e[:ns] = 1

    w0 = w0[:ns + nt]
    wt = w0

    iter = 0
    convA = 10000

    try:
        while convA > epsilon and iter < 100:
            sumA = np.zeros((Xci.shape[1], Xci.shape[1]))
            C = C.astype(int)
            pair_weights = wt[C[:, 0]] * wt[C[:, 1]] * C[:, 2]

            for i in range(Xci.shape[0]):
                vij = Xci[i, :] - Xcj[i, :]
                sumA = sumA + A0 @ np.outer(vij, vij) * pair_weights[i] * C[i, 2]

            At = At - gamma * (beta * sumA + 2 * A0)

            zeta = np.zeros(ns + nt)

            for k in range(Xci.shape[0]):
                i = C[k, 0]
                j = C[k, 1]
                deta_ij = C[k, 2]

                vij = Xci[k, :] - Xcj[k, :]
                vijA = vij @ At.T
                dij = vijA @ vijA.T

                zeta[i] = zeta[i] + wt[j] * dij * deta_ij
                zeta[j] = zeta[j] + wt[i] * dij * deta_ij

            xi = np.sign(np.maximum(0, -wt))
            dev1 = 2 * lamda * (wt - w0)
            dev2 = beta * zeta
            dev3 = sigma * (2 * (wt @ e - ns) * e + wt ** 2 * xi * e)

            w_dev = dev1 + dev2 + dev3
            wt = wt - gammaW * w_dev
            wt[ns:ns + nt] = 1

            if np.any(np.isnan(wt)):
                raise ValueError("NaN encountered in wt")

            convA = norm(At - A0)
            convW = norm(wt - w0)
            A0 = At
            iter += 1

    except (IndexError, ValueError, RuntimeWarning):
        wt = np.zeros_like(w0)
        result = {}
        result['At'] = At
        result['wt'] = wt
        result['w0'] = w0

    result = {}
    result['At'] = At
    result['wt'] = wt
    result['w0'] = w0

    return At, result
