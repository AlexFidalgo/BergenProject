import numpy as np

def generate_random_data(num_models, num_metrics):
    """Simulates ranking for each metric and model"""
    
    # np.random.seed(42)
    data_matrix = np.column_stack([np.random.permutation(np.arange(1, num_models + 1)) 
                                   for _ in range(num_metrics)])
    return data_matrix

def calculate_mae(matrix, current_metric, reference_metric):
    return np.mean(np.abs(matrix[:, current_metric] - matrix[:, reference_metric]))

def calculate_threshold(D):
    # Calculate the threshold as the 20th percentile of MAE values
    return np.percentile(D, 20)

def clustering_algorithm(matrix):
    D = np.zeros((matrix.shape[1], matrix.shape[1]))
    
    # Calculate MAE values for all combinations of error metrics
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[1]):
            D[i, j] = calculate_mae(matrix, i, j)
    
    threshold = calculate_threshold(D)
    # print(f"threshold = {threshold}")
    num_error_metrics = matrix.shape[1]
    clusters = [[1]]  # Initial cluster with the first metric

    for current_metric_index in range(2, num_error_metrics + 1): #for all error metrics
        max_mae_values = np.zeros(len(clusters))

        for cluster_index, current_cluster in enumerate(clusters): #for all existing clusters
            individual_mae_values = np.zeros(len(current_cluster))

            for cluster_metric_index, cluster_metric in enumerate(current_cluster): #for all error metrics in the current cluster
                individual_mae_values[cluster_metric_index] = D[current_metric_index - 1, cluster_metric - 1]

            max_mae_values[cluster_index] = np.max(individual_mae_values) # Maximum MAE for each cluster

        if np.min(max_mae_values) < threshold:
            min_index, = np.where(max_mae_values == np.min(max_mae_values))
            clusters[min_index[0]].append(current_metric_index)
        else:
            clusters.append([current_metric_index])  # Form a new cluster
    
    return clusters

def print_clusters(clusters):
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")

if __name__ == '__main__':
    num_metrics = 38
    num_models = 89
    
    matrix = generate_random_data(num_models, num_metrics)
    
    clusters = clustering_algorithm(matrix)
    
    print_clusters(clusters)
