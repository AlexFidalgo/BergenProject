library(R.matlab)

## Specify working directory (change accordingly)
setwd("C:\Users\AlexFidalgoZamikhows\Projects\EACH\BergenProject\wp3_data\mat_files")

######################################
### FILES WITH CLUSTER INFORMATION

# Load your MATLAB file
cluster_region <- readMat("clusters_regionwise.mat")
# The file "clusters_regionwise.mat" contains the information
# about the clustering for each region

# The object "cluster_region$c101" contains 20 rows (variables) 
# and 8 columns (regions)
nrow(cluster_region$c101)
ncol(cluster_region$c101)
# The file contains 20 variables, but we are working
# with two variables: Precipitation (variable 3)
# and Temperature (variable 10).
# The file contains 8 regions in the following order:
# 1: British Isles (BI) 2: Iberian Peninsula (IP) 3: France (FR)
# 4: Mid-Europe (ME) 5: Scandinavia (SC) 6: Alps (AL) 
# 7: Mediterranean 8: Eastern Europe (EA)

# The object "cluster_region$c101" may be referenced by a list with a total
# of seven indices. 
# The first set of indices indicates the variable and region.
# The second and third indices should be fixed in [[1]]
# For example: 
nrow(cluster_region$c101[3,1][[1]][[1]])
## Indicate the number of clusters for precipitation (variable 3)
## in region 1 (British Isles)
# There are 15 clusters.(as indicated in the paper section 4.2.1, figure 7) 

nrow(cluster_region$c101[10,1][[1]][[1]])
## Indicate the number of clusters for temperature (variable 10)
## in region 1 (British Isles)
# There are 12 clusters.(as indicated in the paper section 4.2.2, figure 9) 

# The fourth set of indices indicates the cluster. There is only on column.
# The fifth and sixth  indices should be fixed in [[1]]
# For example: 
## Indicate the number of error metrics in cluster 1 for precipitation (variable 3)
## in region 1 (British Isles)
nrow(cluster_region$c101[3,1][[1]][[1]][1,1][[1]][[1]])

## Show the error metrics 
cluster_region$c101[3,1][[1]][[1]][1,1][[1]][[1]]

# The seventh set of indices indicates the error metric. There is only on column.
# For example
## Show the first error metric in cluster 1 for precipitation (variable 3)
## in region 1 (British Isles)
cluster_region$c101[3,1][[1]][[1]][1,1][[1]][[1]][1,1]


#######################################################
#######################################################
# CREATING DATAFRAME - BRITISH ISLES - PRECIPITATION
#### Obs.: Similar procedure can be done for other regions, changing the file accordingly

# Load your MATLAB file
mat_data <- readMat("error_ppt_BI.mat")

# Convert the three-dimensional array to a vector
mat_vector <- as.vector(mat_data$ppt)

Metric = rep(seq_len(dim(mat_data$ppt)[3]), each = dim(mat_data$ppt)[1] * dim(mat_data$ppt)[2])
Gridpoint = rep(rep(seq_len(dim(mat_data$ppt)[2]), each = dim(mat_data$ppt)[1]), times = dim(mat_data$ppt)[3])
Model = rep(seq_len(dim(mat_data$ppt)[1]), times = dim(mat_data$ppt)[2] * dim(mat_data$ppt)[3])

database_BI <- cbind(Model,Gridpoint,Metric,mat_vector)
# The column named "mat_vector" contains the values of the error metric
df_BI <- as.data.frame(database_BI)

#### EXAMPLE FOR GRIDPOINT 4 ########
# Using filtering
# Selecting grid point
gridpoint4 <- subset(df_BI, Gridpoint == 4)
# Selecting metric
gp4metric16 <- subset(gridpoint4, Metric == 16)

# Add descriptors of models
#install.packages("readxl")

# Load the readxl package
library(readxl)

# Specify the path to your Excel file
file_path <- "Models_89_test.xlsx"
# Read a specific sheet from the Excel file
data <- read_excel(file_path, sheet = "Sheet1")

# Print the first few rows of the data
print(head(data))

# Merge based on explicitly specified common columns
merged_df <- merge(gp4metric16, data, by.x = "Model", by.y = "Number")

# Print the merged dataframe
print(merged_df)

