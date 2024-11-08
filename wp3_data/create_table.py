import scipy.io
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analyzing_models')))
from RCMs_GCMs import *
from statistical_analysis import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'prediction')))
from LinearRegression import *

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

    region_table = create_region_table(cons)

    cons = rename_gridpoints(cons)

    return cons, region_table

def create_region_table(cons):

    df = cons.copy()

    df['exists'] = 1

    pivot_table = df.pivot_table(index='Gridpoint', columns='region', values='exists', aggfunc='max', fill_value=0)

    return pivot_table

def rename_gridpoints(cons):

    cons['gp_region'] = cons['Gridpoint'].astype(int).astype(str) + '_' + cons['region']
    cons = cons.drop(columns=['region', 'Gridpoint'])

    return cons

import pandas as pd

def check_physical_variable_equivalency(df):
    """
    This function checks if for each row with a specific physical variable ('ppt' or 'tas'),
    an equivalent row with the other physical variable exists for the same Model, Metric, and gp_region.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the columns Model, Metric, gp_region, physical_variable.
    
    """

    df_ppt = df[df['physical_variable'] == 'ppt']
    df_tas = df[df['physical_variable'] == 'tas']
    
    merged_df = pd.merge(df_ppt, df_tas, on=['Model', 'Metric', 'gp_region'], how='outer', suffixes=('_ppt', '_tas'))
    
    missing_ppt = merged_df[merged_df['physical_variable_ppt'].isnull()]
    missing_tas = merged_df[merged_df['physical_variable_tas'].isnull()]

    print("Rows where 'ppt' is missing:\n", missing_ppt)
    print("Rows where 'tas' is missing:\n", missing_tas)

def check_physical_variable_equivalency_disregarding_metric(df):

    df_red = df[['Model', 'mat_vector', 'physical_variable', 'gp_region']].copy()

    df_ppt = df_red[df_red['physical_variable'] == 'ppt']
    df_tas = df_red[df_red['physical_variable'] == 'tas']

    merged_df = pd.merge(df_ppt, df_tas, on=['Model', 'gp_region'], how='outer', suffixes=('_ppt', '_tas'))

    missing_ppt = merged_df[merged_df['physical_variable_ppt'].isnull()]
    missing_tas = merged_df[merged_df['physical_variable_tas'].isnull()]

    print("Rows where 'ppt' is missing:\n", missing_ppt)
    print("Rows where 'tas' is missing:\n", missing_tas)

import pandas as pd

def check_physical_variable_equivalency_disregarding_metric(df):
    """
    This function checks if for each row, the equivalency of 'ppt' and 'tas' based on the 'mat_vector' column.
    If 'mat_vector' is null for both 'ppt' and 'tas' or not null for both, then the rows are considered equivalent.
    If one is null and the other isn't, they are considered non-equivalent.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the columns Model, mat_vector, physical_variable, and gp_region.
    
    """
    
    df_red = df[['Model', 'mat_vector', 'physical_variable', 'gp_region']].copy()

    df_ppt = df_red[df_red['physical_variable'] == 'ppt']
    df_tas = df_red[df_red['physical_variable'] == 'tas']

    merged_df = pd.merge(df_ppt, df_tas, on=['Model', 'gp_region'], how='outer', suffixes=('_ppt', '_tas'))

    # equivalent_rows = merged_df[
    #     ((merged_df['mat_vector_ppt'].isnull()) & (merged_df['mat_vector_tas'].isnull())) |
    #     ((merged_df['mat_vector_ppt'].notnull()) & (merged_df['mat_vector_tas'].notnull()))
    # ]

    non_equivalent_rows = merged_df[
        ((merged_df['mat_vector_ppt'].isnull()) & (merged_df['mat_vector_tas'].notnull())) |
        ((merged_df['mat_vector_ppt'].notnull()) & (merged_df['mat_vector_tas'].isnull()))
    ]

    # print("Equivalent rows:\n", equivalent_rows)
    print("Non-equivalent rows:\n", non_equivalent_rows)

import pandas as pd

def check_physical_variable_equivalency_by_mat_vector_disregarding_metric(df):
    """
    This function checks if for each row, the equivalency of 'ppt' and 'tas' based on the 'mat_vector' column.
    If 'mat_vector' is null for both 'ppt' and 'tas' or not null for both, then the rows are considered equivalent.
    If one is null and the other isn't, they are considered non-equivalent.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the columns Model, mat_vector, physical_variable, and gp_region.
    
    Returns:
    equivalent_rows (pandas.DataFrame): Rows where 'ppt' and 'tas' are equivalent.
    non_equivalent_rows (pandas.DataFrame): Rows where 'ppt' and 'tas' are not equivalent.
    """

    df_red = df[['Model', 'mat_vector', 'physical_variable', 'gp_region']].copy()

    df_ppt = df_red[df_red['physical_variable'] == 'ppt']
    df_tas = df_red[df_red['physical_variable'] == 'tas']

    merged_df = pd.merge(df_ppt, df_tas, on=['Model', 'gp_region'], how='outer', suffixes=('_ppt', '_tas'))

    # equivalent_rows = merged_df[
    #     ((merged_df['mat_vector_ppt'].isnull()) & (merged_df['mat_vector_tas'].isnull())) |
    #     ((merged_df['mat_vector_ppt'].notnull()) & (merged_df['mat_vector_tas'].notnull()))
    # ]

    non_equivalent_rows = merged_df[
        ((merged_df['mat_vector_ppt'].isnull()) & (merged_df['mat_vector_tas'].notnull())) |
        ((merged_df['mat_vector_ppt'].notnull()) & (merged_df['mat_vector_tas'].isnull()))
    ]

    # print("Equivalent rows:\n", equivalent_rows)
    print("Non-equivalent rows:\n", non_equivalent_rows)

def create_contingency_table(df):

    df['mat_vector_missing'] = df['mat_vector'].isna()
    contingency_table = pd.crosstab(df['physical_variable'], df['mat_vector_missing'])

    return contingency_table

def create_corr_table(df):

    pivot_table = df.pivot_table(values='mat_vector', index='gp_region', columns='Model', aggfunc='mean')

    return pivot_table

def plot_error_distribution_by_region(dataframe, error_column='mat_vector', region_column='gp_region', bins=5000, xlim=(-10, 10)):
    """
    Plots histograms for the distribution of model errors by region with adjusted limits.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    error_column (str): The column name representing model errors.
    region_column (str): The column name representing regions.
    bins (int): Number of bins for the histogram.
    xlim (tuple): The x-axis limits for the histogram to remove outliers.

    Returns:
    None: Displays histograms.
    """
    # Get the unique regions
    regions = dataframe[region_column].unique()
    
    # Plot the histogram for each region
    for region in regions:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=dataframe[dataframe[region_column] == region], x=error_column, bins=bins, kde=True)
        plt.title(f"Error Distribution for Region: {region}")
        plt.xlabel('Model Error')
        plt.ylabel('Frequency')
        plt.xlim(xlim)  # Set x-axis limits to focus on a reasonable range
        plt.show()

def fixing_metric(cons):

    # Only Bergen Metric
    df = cons[cons['Metric']==18]
    df.drop(['Metric'], axis=1, inplace=True)

    return df

def insert_models_ids(cons):

    g,r,real = get_models_dfs()

    merged_cons_real = pd.merge(cons, real, left_on='Model', right_on='id', how='left')
    merged_with_gcm = pd.merge(merged_cons_real, g[['id', 'GCM']], on='GCM', how='left', suffixes=('', '_GCM'))
    final_df = pd.merge(merged_with_gcm, r[['id', 'RCM']], on='RCM', how='left', suffixes=('', '_RCM'))

    final_df = final_df[['Model', 'Metric', 'mat_vector', 'physical_variable', 'gp_region', 'id_GCM', 'id_RCM']]

    final_df.to_csv("data_with_models_ids.csv", index=False)

    return final_df

def generate_pivot_table_RCM_x_GCM(cons):
    """
    Generates a pivot table with id_RCM as rows and id_GCM as columns, 
    where each cell contains the count of rows for that specific id_RCM and id_GCM.

    Parameters:
    cons (pd.DataFrame): The input dataframe containing the columns 'id_RCM' and 'id_GCM'.

    Returns:
    pd.DataFrame: The pivot table with id_RCM as rows and id_GCM as columns.
    """
    pivot_table = pd.pivot_table(
        cons, 
        values='mat_vector',  # Any column can be used here to count the occurrences
        index='id_RCM', 
        columns='id_GCM', 
        aggfunc='count',  # Counts the number of occurrences
        fill_value=0  # Fill missing values with 0
    )
    
    return pivot_table

if __name__ == '__main__':

    cons, region_table = create_cons()

    # check_physical_variable_equivalency(cons)
    # check_physical_variable_equivalency_disregarding_metric(cons)
    # check_physical_variable_equivalency_by_mat_vector_disregarding_metric(cons)
    # ct = create_contingency_table(cons)
    # corr_table = create_corr_table(cons)
    # plot_error_distribution_by_region(cons)

    cons = insert_models_ids(cons)

    ### Exploratory Statistics ### 

    ## 1 ##

    # # average_then_pivot
    # rcms_correlations_for_fixed_gcms = calculate_rcm_correlations_for_fixed_gcms_average_then_pivot(cons)
    # visualize_rcms_correlations_for_fixed_gcms(rcms_correlations_for_fixed_gcms, 'average_then_pivot')

    # # granular_correlations
    # rcms_correlations_for_fixed_gcms = calculate_rcm_correlations_for_fixed_gcms_granular_correlations(cons)
    # visualize_rcms_correlations_for_fixed_gcms(rcms_correlations_for_fixed_gcms, 'granular_correlations')

    # # specific metric
    # rcms_correlations_for_fixed_gcms = calculate_rcm_correlations_for_fixed_gcms_specific_metric_variable(cons)
    # visualize_rcms_correlations_for_fixed_gcms(rcms_correlations_for_fixed_gcms, 'specific_metric_variable')

    ## 2 ##

    # heatmap
    # create_average_error_and_std_maps_for_gcms(cons)

    ## 3 ##

    # # correlation for fixed gcm-region
    # correlation_analysis_by_region_and_gcm_granular(cons, region_code='BI') #just for british isles for now

    ## 4 ## 

    # hierarchical clustering
    # hierarchical_clustering(cons, error_metric=1, region='BI') #just for british isles for now

    # hierarchical clustering with fixed gcm 
    # hierarchical_clustering_fixed_gcm(cons, error_metric=1, region='BI', gcm_id=2)

    ## 5 ##
    averaged_correlation_matrix, averaged_counts_matrix = calculate_general_rcm_correlations(cons)
    visualize_general_rcm_correlations(averaged_correlation_matrix, folder_name='general_rcm_correlations')
    visualize_pair_counts(averaged_counts_matrix)



    ### Prediction ###

    # Linear Regression #

    # for fixed error metric
    # model, coef_df = predict_rcm_error_with_linear_regression_for_fixed_metric(cons, target_rcm=6, fixed_gcm=3, error_metric=1)


    # RCM_GCM_matrix = generate_pivot_table_RCM_x_GCM(cons)


