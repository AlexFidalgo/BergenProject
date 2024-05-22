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

def read_mat_with_h5py(file):

    mat_files_folder = os.path.join(get_current_dir(), 'mat_files')
    f = h5py.File(os.path.join(mat_files_folder, file+'.mat'),'r')
    # data = f.get('data/variable1')
    # data = np.array(data) # For converting to a NumPy array

# clusters_regionwise = read_mat_with_scipy('clusters_regionwise')
# gr_europe = read_mat_with_scipy('gr_Europe')

a = read_mat_with_h5py('clusters_regionwise')

x = 1