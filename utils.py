import numpy as np

def Update(Z, r):
    Z_bar = Z-Z.mean(axis=1)[:,np.newaxis]
    SVD_Z = np.linalg.svd(Z_bar, full_matrices=False)
    Q = SVD_Z[2].T
    if Q.shape[1] < r:
        Q = gram_schmidt(np.c_[Q, np.ones((Q.shape[0], r - Q.shape[1]))])
    P = np.linalg.svd(np.dot(Z_bar, Z_bar.T))[0]
    Z_new = np.sqrt(Z.shape[1]) * np.dot(P, Q.T)
    return Z_new

def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q

def K(x, y):
    return x if np.abs(x) > 10e-8 else y

def cal_bin_dis(a, b):
    #return bin(a ^ b).count('1')
    return (a ^ b)

def index2hash(index, bits):
    hash_code = ""
    bits = bits - 1
    while bits >= 0:
        bit = int(index) // int(2**bits)
        index -= 2**bits * bit
        hash_code = hash_code + str(bit)
        bits -= 1
    return hash_code
