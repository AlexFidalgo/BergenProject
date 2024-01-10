import numpy as np

def generate_random_data(num_models, num_metrics):
    
    """ Simulates ranking for each metric and model"""
    np.random.seed(42)
    return np.random.rand(num_models, num_metrics)

def calculate_mae(matrix, current_metric, reference_metric):
    
    return np.mean(np.abs(matrix[:, current_metric - 1] - matrix[:, reference_metric - 1]))

def clustering_algorithm(matrix, threshold):
    
    num_metrics = matrix.shape[1]
    c = [[1]]  # Initial cluster with the first metric

    for i in range(2, num_metrics + 1):
        d2 = np.zeros(len(c))

        for i1, cluster in enumerate(c):
            d1 = np.zeros(len(cluster))

            for i2, el in enumerate(cluster):
                d1[i2] = calculate_mae(matrix, i - 1, el)  # Calculate MAE

            d2[i1] = np.max(d1)  # Maximum MAE for each cluster

        if np.min(d2) < threshold:
            l1, = np.where(d2 == np.min(d2))
            c[l1[0]].append(i)
        else:
            c.append([i])  # Form a new cluster
    
    return c

def print_clusters(clusters):
    
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")

if __name__ == '__main__':
    
    threshold = 0.35
    num_metrics = 38
    num_models = 89
    
    data_matrix = generate_random_data(num_models, num_metrics)
    
    clusters = clustering_algorithm(data_matrix, threshold)
    
    print_clusters(clusters)