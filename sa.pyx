import numpy as np
cimport numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix
cimport cython

DTYPE = np.double

ctypedef np.float64_t DTYPE_t



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def Jcython(np.ndarray[DTYPE_t, ndim=2]X, np.ndarray[DTYPE_t, ndim=2]y):

    cdef:
        Py_ssize_t nrow = X.shape[0]
        Py_ssize_t nrow_y = y.shape[0]
        Py_ssize_t ncol = X.shape[1]
        Py_ssize_t ii, jj, kk

        np.ndarray[DTYPE_t, ndim=2] D = np.zeros((nrow, nrow_y), dtype=DTYPE)
        double cost = 0
        double min_value, tmpss, diff


    for ii in range(nrow):
        min_value = 10000000000
        for jj in range(nrow_y):
            tmpss = 0
            for kk in range(ncol):
                diff = X[ii, kk] - y[jj, kk]
                tmpss += diff * diff
            D[ii, jj] = tmpss
            if D[ii, jj] < min_value:
                min_value = D[ii, jj]
        cost += min_value/nrow

    return cost


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def J(np.ndarray[DTYPE_t, ndim=2]X, np.ndarray[DTYPE_t, ndim=2]y):
    cdef int row = X.shape[0]
    cdef int col = y.shape[0]
    cdef double cost = 0
    cdef np.ndarray[DTYPE_t, ndim=2]d = np.zeros([row, col], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1]min_dist = np.zeros(row, dtype=DTYPE)
    d = distance_matrix(X, y)**2
    min_dist = np.min(d, axis=1)
    cost = np.sum(min_dist)/row
    return cost

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def SA(np.ndarray[DTYPE_t, ndim=2] X, double T0, int kmax, int n_centroids, int N, int mean_start):

    cdef double T = 0
    cdef double j_y = 0
    cdef double j_y_hat = 0
    cdef int total = N*kmax
    # cdef int pos = 0
    cdef double eps = 0.001
    cdef double r = 0
    cdef double min_atual = 10000000
    cdef int dimension = X.shape[1]
    cdef int rows = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] y = np.zeros([n_centroids, dimension], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] y_hat = np.zeros([n_centroids, dimension], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] pert = np.random.normal(size=(n_centroids, dimension))
    cdef np.ndarray[DTYPE_t, ndim=2] Js = np.zeros([total, 2], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] y_min = np.zeros([n_centroids, dimension], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] Jmin = np.zeros([total, 2], dtype=DTYPE)
    cdef Py_ssize_t k, i, j, pos

    cdef DTYPE_t value

    if mean_start > 0:
        ##calcula valor médio em cada dimensao
        for i in range(n_centroids):
            for j in range(dimension):
                for k in range(rows):
                    y[i,j] += X[k, j] / rows
    else:
        y = np.random.normal(size=(n_centroids, dimension))

    for k in tqdm(range(kmax)):
        T = T0/(np.log2(2+k))
        for i in range(N):
            pert = np.random.normal(size=(n_centroids, dimension))
            y_hat = y + eps*pert
            
            j_y = Jcython(X, y)
            j_y_hat = Jcython(X, y_hat)
            deltaJ = np.exp((j_y - j_y_hat)/T)
            r = np.random.random()
            pos = k * N + i
            if r < deltaJ:
                y = y_hat
                Js[pos, 0] = T
                Js[pos, 1] = j_y_hat
            else:
                Js[pos, 0] = T
                Js[pos, 1] = j_y
            
            if j_y_hat < min_atual:
                Jmin[pos, 0] = T
                Jmin[pos, 1] = j_y_hat
                min_atual = j_y_hat
                y_min = y_hat
            else:
                Jmin[pos, 0] = T
                Jmin[pos, 1] = min_atual
    return Js, Jmin, y_min, T


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def FSA(np.ndarray[DTYPE_t, ndim=2] X, double T0, int kmax, int n_centroids, int N, int mean_start):

    cdef double T = 0
    cdef double j_y = 0
    cdef double j_y_hat = 0
    cdef int total = N*kmax
    # cdef int pos = 0
    cdef double eps = 0.001
    cdef double r = 0
    cdef double min_atual = 10000000
    cdef int dimension = X.shape[1]
    cdef int rows = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] y = np.zeros([n_centroids, dimension], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] y_hat = np.zeros([n_centroids, dimension], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] pert = np.random.standard_cauchy(size=(n_centroids, dimension))
    cdef np.ndarray[DTYPE_t, ndim=2] Js = np.zeros([total, 2], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] y_min = np.zeros([n_centroids, dimension], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] Jmin = np.zeros([total, 2], dtype=DTYPE)
    cdef Py_ssize_t k, i, j, pos

    cdef DTYPE_t value

    if mean_start > 0:
        ##calcula valor médio em cada dimensao
        for i in range(n_centroids):
            for j in range(dimension):
                for k in range(rows):
                    y[i,j] += X[k, j] / rows
    else:
        y = np.random.standard_cauchy(size=(n_centroids, dimension))

    for k in tqdm(range(kmax)):
        T = T0/(1+k)
        for i in range(N):
            pert = np.random.standard_cauchy(size=(n_centroids, dimension))
            y_hat = y + eps*pert
            
            j_y = Jcython(X, y)
            j_y_hat = Jcython(X, y_hat)
            deltaJ = np.exp((j_y - j_y_hat)/T)
            r = np.random.random()
            pos = k * N + i
            if r < deltaJ:
                y = y_hat
                Js[pos, 0] = T
                Js[pos, 1] = j_y_hat
            else:
                Js[pos, 0] = T
                Js[pos, 1] = j_y
            
            if j_y_hat < min_atual:
                Jmin[pos, 0] = T
                Jmin[pos, 1] = j_y_hat
                min_atual = j_y_hat
                y_min = y_hat
            else:
                Jmin[pos, 0] = T
                Jmin[pos, 1] = min_atual
    return Js, Jmin, y_min, T