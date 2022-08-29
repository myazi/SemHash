import sys
import numpy as np
from sklearn.neighbors import DistanceMetric

def eval_retrieval(vector, label, top_n=100) :
    distance = DistanceMetric.get_metric("hamming").pairwise(vector, vector)
    np.fill_diagonal(distance, np.Infinity)
    sort_idx = np.argsort(distance, axis=-1)
    predict = []
    for i in range(top_n) :
        predict.append(
            (np.equal(label[sort_idx[:, i].reshape(-1)], label)).reshape(-1, 1)
            )
    predict = np.concatenate(predict, axis=1)
    precision = np.mean(np.sum(predict, axis=1)/top_n)
    precision = round(precision, 4)
    print("Top {} precision : {}".format(top_n, precision))
    return precision

