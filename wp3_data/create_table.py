import scipy.io
import os
import numpy as np
import pandas as pd

def get_current_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return current_dir

def read_mat_with_scipy(file):
    mat_files_folder = os.path.join(get_current_dir(), 'mat_files')
    return scipy.io.loadmat(os.path.join(mat_files_folder, file +'.mat'))

def read_excel(file, sheet_name = None):
    current_dir = get_current_dir()
    file_path = os.path.join(get_current_dir(), 'mat_files', file)
    df = pd.read_excel(file_path, sheet_name)
    return df

def get_variable_region_dataframe(variable, region):

    mat_data = read_mat_with_scipy(f"error_{variable}_{region}")[variable]
    mat_vector = mat_data.flatten(order='F')

    dim1, dim2, dim3 = mat_data.shape

    Metric = np.repeat(np.arange(1, dim3 + 1), dim1 * dim2)
    Gridpoint = np.tile(np.repeat(np.arange(1, dim2 + 1), dim1), dim3) 
    Model = np.tile(np.arange(1, dim1 + 1), dim2 * dim3)

    database = np.column_stack((Model, Gridpoint, Metric, mat_vector))

    df = pd.DataFrame(database, columns=['Model', 'Gridpoint', 'Metric', 'mat_vector'])

    return df

def get_regions(): 

    return ['AL', 'BI', 'EA', 'FR', 'IP', 'MD', 'ME', 'SC']

def get_variables():

    return ['ppt', 'tas']

def create_cons():

    cons = pd.DataFrame()

    regions = get_regions()

    variables = get_variables()

    for variable in variables:

        for region in regions:
        
            df = get_variable_region_dataframe(variable, region)

            df['physical_variable'] = variable

            df['region'] = region

            cons = pd.concat([cons, df])

    return cons

def create_region_table(cons):

    df = cons.copy()

    df['exists'] = 1

    pivot_table = df.pivot_table(index='Gridpoint', columns='region', values='exists', aggfunc='max', fill_value=0)

    return pivot_table



if __name__ == '__main__':

    cons = create_cons()

    region_table = create_region_table(cons)

    x