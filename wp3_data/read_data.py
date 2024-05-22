import scipy.io
import os
import numpy as np
import h5py

def get_current_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return current_dir

def read_mat_with_scipy(file):

    mat_files_folder = os.path.join(get_current_dir(), 'mat_files')
    return scipy.io.loadmat(os.path.join(mat_files_folder, 'clusters_regionwise'+'.mat'))

# def read_mat_with_h5py(file):

#     mat_files_folder = os.path.join(get_current_dir(), 'mat_files')
#     f = h5py.File(os.path.join(mat_files_folder, file+'.mat'),'r')
#     # data = f.get('data/variable1')
#     # data = np.array(data) # For converting to a NumPy array
# a = read_mat_with_h5py('clusters_regionwise')

cluster_region = read_mat_with_scipy("clusters_regionwise") # The file "clusters_regionwise.mat" contains the information about the clustering for each region

cluster_region101 = cluster_region['c101']

print(f"len(cluster_region101[2][0]): {len(cluster_region101[2][0])}") # number of clusters for precipitation (3rd variable, index 2) in the 1st region (index 0)

print(f"len(cluster_region101[9][0]): {len(cluster_region101[9][0])}") # number of clusters for temperature (10th variable, index 9) in the 1st region
