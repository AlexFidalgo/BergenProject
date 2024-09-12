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

# cluster_region <- readMat("clusters_regionwise.mat")
cluster_region = read_mat_with_scipy("clusters_regionwise") # The file "clusters_regionwise.mat" contains the information about the clustering for each region

# cluster_region$c101
cluster_region101 = cluster_region['c101'] # contains 20 rows (variables) and 8 columns (regions); 3rd variable is precipitation, 10th variable is temperature

print(f"\nShape: {cluster_region101.shape}")

# cluster_region101 can be by a list with a total of 7 indices in R

# nrow(cluster_region$c101[3,1][[1]][[1]])
print(f"\nlen(cluster_region101[2][0]): {len(cluster_region101[2][0])}") # number of clusters for precipitation (3rd variable, index 2) in the 1st region (index 0)

# nrow(cluster_region$c101[10,1][[1]][[1]])
print(f"\nlen(cluster_region101[9][0]): {len(cluster_region101[9][0])}") # number of clusters for temperature (10th variable, index 9) in the 1st region (index 0)

# nrow(cluster_region$c101[3,1][[1]][[1]][1,1][[1]][[1]])
print(f"\nlen(cluster_region101[2][0][0][0]): {len(cluster_region101[2][0][0][0])}") # number of error metrics in the 1st cluster (index 0) for the 3rd variable (precipitation) (index 3) in the first region (index 0)

# cluster_region$c101[3,1][[1]][[1]][1,1][[1]][[1]]
print(f"\nError metrics: {cluster_region101[2][0][0][0]}") # shows the error metrics

# cluster_region$c101[3,1][[1]][[1]][1,1][[1]][[1]][1,1]
print(f"\nFirst error metric: {cluster_region101[2][0][0][0][0]}") # shows the first (index 0) error metric


### CREATING DATAFRAME - BRITISH ISLES - PRECIPITATION

# mat_data <- readMat("error_ppt_BI.mat")
# mat_vector <- as.vector(mat_data$ppt)
mat_data = read_mat_with_scipy("error_ppt_BI")
mat_data_ppt = mat_data['ppt']
mat_vector = mat_data_ppt.flatten(order='F')

# Metric = rep(seq_len(dim(mat_data$ppt)[3]), each = dim(mat_data$ppt)[1] * dim(mat_data$ppt)[2])
# Gridpoint = rep(rep(seq_len(dim(mat_data$ppt)[2]), each = dim(mat_data$ppt)[1]), times = dim(mat_data$ppt)[3])
# Model = rep(seq_len(dim(mat_data$ppt)[1]), times = dim(mat_data$ppt)[2] * dim(mat_data$ppt)[3])
dim1, dim2, dim3 = mat_data_ppt.shape
Metric = np.repeat(np.arange(1, dim3 + 1), dim1 * dim2) # creates a 1D array by repeating each integer in the range from 1 to dim3 (inclusive) dim1 * dim2 times
Gridpoint = np.tile(np.repeat(np.arange(1, dim2 + 1), dim1), dim3) # creates a 1-dimensional numpy array by repeating each integer in the range from 1 to dim2 (inclusive) dim1 times, and then repeating this entire sequence dim3 times
Model = np.tile(np.arange(1, dim1 + 1), dim2 * dim3) # creates a 1-dimensional numpy array (Model) by repeating the sequence of integers from 1 to dim1 (inclusive) dim2 * dim3 times

# database_BI <- cbind(Model,Gridpoint,Metric,mat_vector)
# df_BI <- as.data.frame(database_BI)
database_BI_np = np.column_stack((Model, Gridpoint, Metric, mat_vector))
df_BI = pd.DataFrame(database_BI_np, columns=['Model', 'Gridpoint', 'Metric', 'mat_vector']) #The column named "mat_vector" contains the values of the error metric

### Example for Gripoint 4

# gridpoint4 <- subset(df_BI, Gridpoint == 4)
gridpoint4 = df_BI[df_BI['Gridpoint'] == 4] 

# gp4metric16 <- subset(gridpoint4, Metric == 16)
gp4metric16 = gridpoint4[gridpoint4['Metric'] == 16]

# library(readxl)
# file_path <- "Models_89_test.xlsx"
# data <- read_excel(file_path, sheet = "Sheet1")
file_path = "Models_89_test.xlsx"
data = read_excel(file_path, sheet_name="Sheet1")

# merged_df <- merge(gp4metric16, data, by.x = "Model", by.y = "Number")
merged_df = pd.merge(gp4metric16, data, left_on="Model", right_on="Number")