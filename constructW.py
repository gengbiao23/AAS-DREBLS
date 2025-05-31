from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def constructW(fea, options):
    """
    Constructs a weight matrix W using KNN or supervised methods.
    fea: Feature matrix (n_samples x n_features)
    options: Dictionary containing various options, such as:
        - NeighborMode: 'KNN' or 'Supervised'
        - k: number of nearest neighbors (only for KNN)
        - gnd: ground truth labels (only for Supervised)
        - WeightMode: 'HeatKernel' (default)
        - t: heat kernel parameter (only for HeatKernel)
    """
    n_samples = fea.shape[0]
    if options['NeighborMode'] == 'KNN':
        k = options.get('k', 5)
        # 计算欧几里得距离
        dist = euclidean_distances(fea, fea)
        # 对每行进行排序，取前k个最近邻
        sorted_indices = np.argsort(dist, axis=1)[:, 1:k+1]
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            W[i, sorted_indices[i]] = 1
        return W
    elif options['NeighborMode'] == 'Supervised':
        # Supervised mode: construct graph based on ground truth labels (gnd)
        gnd = options['gnd']
        n_label = len(np.unique(gnd))
        W = np.zeros((n_samples, n_samples))
        for i in range(n_label):
            class_idx = np.where(gnd == i)[0]
            dist = euclidean_distances(fea[class_idx], fea[class_idx])
            W[class_idx[:, None], class_idx] = np.exp(-dist / (2 * options['t'] ** 2))
        return W
    else:
        raise ValueError('Invalid NeighborMode. Supported modes are KNN and Supervised.')
